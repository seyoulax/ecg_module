import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from utils import create_conf_interval

class CatBoostModel:
  def __init__(self, seed=42, device="CPU"):

    """data params"""
    self.data = None
    self.train_data = None
    self.val_data = None
    self.test_data = None

    """global params"""
    self.model = None
    self.best_score = None
    self.best_model = None

    self.device = device
    self.seed = seed

    self.fitted_model = False
    self.loaded_model = False

    """optuna params"""
    self.best_trial = None
    self.best_optuna_model = None
    self.best_optuna_score = None


  def init_data(self, data, target_column = "target", splits=[.8, .15, .05], shuffle_train=True, shuffle_valid=False, embedding_features=[]):

    assert(np.sum(splits) == 1), "incorrect splits values, sum must be equal to 1"

    self.data = data

    X = data.drop(columns=[target_column])
    if type(target_column) == str:
      y = data.loc[:, target_column]
    else:
      y = data.iloc[:, target_column]

    y = np.array(y, dtype=np.int8)
    X = np.array(X, dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(splits[1] + splits[2]), random_state=self.seed, shuffle=shuffle_train)

    self.train_data = Pool(data=X_train, label=y_train.reshape(-1, 1), embedding_features=embedding_features)

    if len(splits) == 3:
      X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=(splits[2] / (splits[1] + splits[2])), random_state=self.seed, shuffle=shuffle_valid)
      self.test_data = Pool(data=X_test, label=y_test.reshape(-1, 1), embedding_features=embedding_features)

    self.val_data = Pool(data=X_val, label=y_val.reshape(-1, 1), embedding_features=embedding_features)


  def init_model(self, parameters={}, task="classification"):

    self.fitted_model = False

    if task == "classification":
      self.model = CatBoostClassifier(**parameters, random_seed=self.seed, task_type=self.device)

    else:
      self.model = CatBoostRegressor(**parameters, random_seed=self.seed, task_type=self.device)

  def load_model(self, path):

    self.init_model()
    self.model.load_model(path)
    self.fitted_model = True

  def fit_model(self, parameters={}):

    if self.train_data is None:
      raise Exception("Training data can not be None")

    if self.val_data is None:
      raise Exception("Validation data can not be None")

    self.model.fit(X=self.train_data, eval_set=self.val_data, **parameters)

    self.fitted_model = True

    print("Training Done")

  #inf_data в формате [X, y]
  def inference_model(self, inf_data=None, predict_proba=False, evaluate_metrics=["f1_score", "accuracy_score"], main_metric="f1_score", task_type="GPU"):

    assert(self.fitted_model), "there is no trained model"
    assert(main_metric in evaluate_metrics), "there is no trained model"

    data = self.test_data if inf_data is None else inf_data

    if not predict_proba:
      predictions = self.model.predict(data, task_type=self.device)

    else:
      predictions = self.model.predict_proba(data, task_type=self.device)
      return predictions

    metrics = {key:0 for key in evaluate_metrics}
    for metric_name in evaluate_metrics:
      metrics[metric_name] = getattr(sklearn.metrics, metric_name)(data.get_label(), predictions)

    if self.best_score is None or self.best_score < metrics[main_metric]:
      self.best_score = metrics[main_metric]
      self.best_model = self.model

    return metrics

  #editable
  def optuna_objective(self, metric_name, trial):

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "task_type" : self.device
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    optuna_gbm = CatBoostClassifier(**param)

    optuna_gbm.fit(self.train_data, eval_set=self.val_data, verbose=0, early_stopping_rounds=100)

    predictions = optuna_gbm.predict(self.test_data)
    metric = getattr(sklearn.metrics, metric_name)(self.test_data.get_label(), predictions)

    if self.best_optuna_score is None or self.best_optuna_score < metric:
      self.best_optuna_score = metric
      self.best_optuna_model = optuna_gbm

    return metric

  def tune_params(self, metric_name="f1_score", direction="maximize", n_trials=10, timeout_study=1000, task_type="CPU", show_progress_bar=True):
    study = optuna.create_study(direction=direction)
    study.optimize(lambda x: self.optuna_objective(metric_name, x), n_trials=n_trials, timeout=timeout_study, show_progress_bar=show_progress_bar)

    self.best_trial = study.best_trial

    print(f"Count trial: {len(study.trials)}")

  def get_best_params_optuna(self):

    assert(self.best_trial is not None), "No study have been done"

    trial = self.best_trial

    print(f"Best metric value: {trial.value}")

    print("Best trial params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}", end="\n")


def create_conf_interval_gbm(gbm, num_boostrapping):
  preds = gbm.model.predict(gbm.test_data)
  target = gbm.test_data.get_label()

  return create_conf_interval(target, preds, num_boostrapping)

