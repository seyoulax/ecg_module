import torch
import numpy as np
import pandas as pd
import sklearn
import random
import optuna

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
from utils import create_conf_interval
from ecgnet_model import ECGNet

class ECGNetWorker():
    class MetricMonitor:
        def __init__(self, float_precision=3):
            self.float_precision = float_precision
            self.reset()

        def reset(self):
            self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

        def update(self, metric_name, val):
            metric = self.metrics[metric_name]

            metric["val"] += val
            metric["count"] += 1
            metric["avg"] = metric["val"] / metric["count"]

        def __str__(self):
            return " | ".join(
                [
                    "{metric_name}: {avg:.{float_precision}f}".format(
                        metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                    )
                    for (metric_name, metric) in self.metrics.items()
                ]
            )

    class EcgPTBDataset(Dataset):
        def __init__(self, labels=[], path='/'):
            self.x_paths = [labels.iloc[i, 0] for i in range(len(labels))]
            self.labels = [labels.iloc[i, 1] for i in range(len(labels))]
            self.path = path

        def __len__(self):
            return len(self.x_paths)

        def __getitem__(self, idx):
            hr = torch.tensor(np.load(self.path + self.x_paths[idx] + '.npy'))[None, :, :]

            target = self.labels[idx]

            return hr, target

    def __init__(self, device="cpu"):

        self.device = torch.device(device)

        self.labels = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.model = None

        self.fitted_model = False

    def evaluate_data_for_ovr(self, labels, target_class):

        left_classes = [i for i in labels.result_class.unique() if i != target_class]
        num_others = (len(labels[labels.result_class == target_class]) * 2) // 15
        data = labels[labels.result_class == target_class]
        data.loc[:, ["result_class"]] = 1
        data.index = range(0, len(data))
        for cur_class in left_classes:
            cur_class_data = labels[(labels.result_class == cur_class)]
            cur_class_data = cur_class_data[
                ~cur_class_data.record_name.isin(labels[labels.result_class != cur_class].record_name)]
            cur_frame = cur_class_data.sample(n=min(len(cur_class_data), num_others))
            cur_frame.loc[:, ["result_class"]] = 0
            data = pd.concat([data, cur_frame], axis=0)

        self.labels = data

        print("evaluate done")

    def create_torch_dataset(self, ecg_path=None, splits=[.8, .15, .05]):

        assert (np.sum(splits) == 1), "incorrect splits values, sum must be equal to 1"

        dataset = self.EcgPTBDataset(self.labels, ecg_path)

        if len(splits) == 3:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, lengths=splits)
        else:
            self.train_dataset, self.val_dataset = random_split(dataset, lengths=splits)

        print(f"train set len: {len(self.train_dataset)}, val set len: {len(self.val_dataset)}")

    def create_model(self, params={}):

        self.fitted_model = False

        try:
            model = ECGNet(**params)
            self.model = model.to(self.device)
            print("model succesfully build")

        except:
            raise Exception("error occured during model building")

    def load_model(self, path=False, params={}):

        model = ECGNet(**params)
        model = model.to(self.device)

        model.load_state_dict(state_dict=torch.load(path, map_location=self.device))

        self.model = model

    def train_model(self, n_epochs=10, checkpoints=None, checkpoints_strategy="max", loss="BCELoss", optimizer="Adam",
                    lr=1e-3, scheduler="ReduceLROnPlateau", eval_metric="f1_score", num_workers=1, batch_size=64,
                    shuffle_train=True, shuffle_val=False):

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=0 if self.device else num_workers,
                                  pin_memory=True if self.device else False)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle_val,
                                num_workers=0 if self.device else num_workers,
                                pin_memory=True if self.device else False)

        loss_fn = getattr(torch.nn, loss)()
        optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        scheduler = getattr(torch.optim.lr_scheduler, scheduler)(optimizer)

        n_epochs = n_epochs

        max_score = -1
        best_epoch = 1

        for epoch in range(n_epochs):
            self.train_epoch(train_loader, loss_fn, eval_metric, optimizer, epoch)
            val_score, loss_v = self.validate_epoch(val_loader, loss_fn, eval_metric, epoch)
            scheduler.step(val_score)

            if checkpoints is not None:

                stmt = False

                if checkpoints_strategy == "max":
                    stmt = max_score < val_score
                elif type(checkpoints_strategy) == float:
                    stmt = val_score >= checkpoints_strategy

                if stmt:
                    torch.save(self.model.state_dict(), f'{checkpoints}/score:{val_score}.pth')

            if max_score < val_score:
                best_epoch = epoch
                max_score = val_score

        self.fitted_model = True
        print(f"Training done, best epoch: {best_epoch}, best score: {max_score}")

    def train_epoch(self, train_loader, loss_fn, eval_metric, optimizer, epoch):
        metric_monitor = self.MetricMonitor(float_precision=4)
        self.model.train()
        metric = getattr(sklearn.metrics, eval_metric)
        stream = tqdm(train_loader)
        for i, batch in enumerate(stream, start=1):
            x_batch, y_batch = batch
            y_batch = y_batch.to(self.device, non_blocking=True)
            x_batch = x_batch.to(self.device, non_blocking=True)
            output = self.model(x_batch.float()).view(1, -1)[0]
            loss = loss_fn(output, y_batch.float())
            output = (output > 0.5).to(torch.int32)
            score = metric(y_batch.cpu(), output.cpu())
            metric_monitor.update("Loss", loss)
            metric_monitor.update(eval_metric, score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stream.set_description(f"Epoch: {epoch}. Train.  {metric_monitor}")

    def validate_epoch(self, val_loader, loss_fn, eval_metric, epoch):
        metric_monitor = self.MetricMonitor(float_precision=4)
        self.model.eval()
        metric = getattr(sklearn.metrics, eval_metric)
        stream = tqdm(val_loader)
        with torch.no_grad():
            for i, batch in enumerate(stream, start=1):
                x_batch, y_batch = batch
                y_batch = y_batch.to(self.device, non_blocking=True)
                x_batch = x_batch.to(self.device, non_blocking=True)
                output = self.model(x_batch.float()).view(1, -1)[0]
                loss = loss_fn(output, y_batch.float())
                output = (output > 0.5).to(torch.int32)
                score = metric(y_batch.cpu(), output.cpu())
                metric_monitor.update("Loss", loss)
                metric_monitor.update(eval_metric, score)
                stream.set_description(f"Epoch: {epoch}. Validation. {metric_monitor}")
        return metric_monitor.metrics[eval_metric]["avg"], metric_monitor.metrics["Loss"]["avg"]

    def inference_model(self, eval_metric="f1_score"):

        targets = []
        preds = []
        self.model.eval()
        metric = getattr(sklearn.metrics, eval_metric)

        with torch.no_grad():
            for hr, target in self.test_dataset:
                targets.append(target)
                hr = hr.to(self.device).unsqueeze(0)
                output = self.model(hr.float()).view(1, -1)[0]
                output = (output > 0.5).to(torch.int32)
                preds.append(output.cpu().item())

        score = metric(np.array(targets), np.array(preds))

        print(f"len test set: {len(targets)}, score: {score}")
        return score

    def optuna_objective(self, trial):

        lr_base = trial.suggest_float("lr_base", 1e-3, 1e-2)
        optimizer = trial.suggest_categorical("optimizer", ["Adam", "Adagrad", "RMSprop"])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        embedding_size = trial.suggest_int("embedding_size", 200, 400)
        # dropout = trial.suggest_categorical("dropout", [False, 0.1])

        self.create_model(params={"embedding_size": embedding_size, "dropout": False, "num_layers": num_layers})

        self.train_model(n_epochs=5, checkpoints=None, optimizer=optimizer, lr=lr_base, batch_size=256)

        metric = self.inference_model()
        print(metric)

        return metric

    def tune_model(self, n_trials=None):

        study = optuna.create_study(direction="maximize")
        study.optimize(self.optuna_objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_trial

def create_conf_interval_ecgnet(worker, metric, num_bootstraping):
  targets = []
  preds = []
  worker.model.eval()
  metric = getattr(sklearn.metrics, metric)

  with torch.no_grad():
    for hr, target in worker.test_dataset:
      targets.append(target)
      hr = hr.to(worker.device).unsqueeze(0)
      output = worker.model(hr.float()).view(1, -1)[0]
      output = (output > 0.5).to(torch.int32)
      preds.append(output.cpu().item())

  return create_conf_interval(targets, preds, num_bootstraping)