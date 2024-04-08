import random
import torch
import numpy as np
import pandas as pd
import torch
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from time_frequency_domain_features import *


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def count_time_features(data):

  result_mean_fea = mean_fea(data)
  result_rms_fea = rms_fea(data)
  result_sr_fea = sr_fea(data)
  result_am_fea = am_fea(data)
  result_skew_fea = skew_fea(data)
  result_kurt_fea = kurt_fea(data)
  result_max_fea = max_fea(data)
  result_min_fea = min_fea(data)
  result_pp_fea = pp_fea(data)
  result_var_fea = var_fea(data)
  result_waveform_index_fea = waveform_index(data)
  result_peak_index_fea = peak_index(data)
  result_impluse_factor_fea = impluse_factor(data)

  return np.array([
      result_mean_fea,
      result_rms_fea,
      result_var_fea,
      result_waveform_index_fea,
      result_peak_index_fea,
      result_sr_fea,
      result_am_fea,
      result_skew_fea,
      result_kurt_fea,
      result_max_fea,
      result_min_fea,
      result_pp_fea,
      result_impluse_factor_fea
  ])

#count all frequency features
def count_freq_features(data):

  result_fft_mean = fft_mean(data)
  result_fft_var = fft_var(data)
  result_fft_std = fft_std(data)
  result_fft_entropy = fft_entropy(data)
  result_fft_energy = fft_energy(data)
  result_fft_skew = fft_skew(data)
  result_fft_kurt = fft_kurt(data)
  result_fft_shape_mean = fft_shape_mean(data)
  result_fft_shape_std = fft_shape_std(data)

  return np.array([
      result_fft_mean,
      result_fft_var,
      result_fft_std,
      result_fft_entropy,
      result_fft_energy,
      result_fft_skew,
      result_fft_kurt,
      result_fft_shape_mean,
      result_fft_shape_std
  ])

### extracting all features
def get_all_features(ecg_signal):


    tmp_raw = np.array(ecg_signal)
    time_features_tmp=[]
    fre_features_tmp=[]

    for j in range(0,12):

        tmp_lead = tmp_raw[j]
        result_time_features = count_time_features(tmp_lead)
        result_fre_features = count_freq_features(tmp_lead)
        time_features_tmp.append(result_time_features)
        fre_features_tmp.append(result_fre_features)

    return np.array(time_features_tmp), np.array(fre_features_tmp)

embeds = []
def make_dataset(model=False, data=False, root_path=False, device=False, len_embeds=False, len_features=False, save_path=False, step=1):
  global embeds

  def get_signal_embeds(model, loader, device):

    model.eval()
    model = model.to(device)
    all_embeds = np.array([])
    print("getting embeds")
    for x in tqdm(loader):
      x = x.to(device)
      with torch.no_grad():
        all_embeds = np.append(all_embeds, model.embed(x.float()).cpu().flatten().numpy())

    return all_embeds

  device = torch.device(device)

  x_paths = [data.iloc[i, 0] for i in range(len(data))]
  hrs = [torch.tensor(np.load(root_path + x_path + '.npy'))[None, :, :] for x_path in x_paths]
  loader = DataLoader(hrs, batch_size=256, shuffle=False)
  embeds = get_signal_embeds(model, loader, device)
  embeds = pd.DataFrame(embeds.reshape(-1, len_embeds), columns=[f"e_{i}" for i in range(0, len_embeds)])

  labels = [data.iloc[i, 1] for i in range(len(data))]
  cb_dataset = np.zeros((1, len_features + 1))

  print("making dataset")
  for i in tqdm(range(len(hrs))):
    t_f, f_f = get_all_features(hrs[i][0])
    features = np.append(t_f, f_f)
    cb_dataset = np.append(cb_dataset, np.append(np.array(labels[i]), features).reshape(1, -1))

  cb_dataset_reshaped = np.reshape(cb_dataset, (-1, + len_features + 1))

  cb_dataset_reshaped = pd.DataFrame(cb_dataset_reshaped, columns=["target"]+[f"f_{i}" for i in range(0, len_features)])
  cb_dataset_reshaped = cb_dataset_reshaped.drop(index=[0])

  cb_dataset_reshaped = pd.concat([cb_dataset_reshaped, embeds], axis=1)

  if save_path:
    cb_dataset_reshaped.to_csv(f"{save_path}/dataset_cbm.csv", index=False)

  return cb_dataset_reshaped

def create_conf_interval(target, predictions, num_bootstraping):

  target = np.array(target)
  predictions = np.array(predictions)

  acc_test = np.mean(predictions == target)

  rng = np.random.RandomState(seed=12345)
  idx = np.arange(target.shape[0])
  test_accuracies = []

  for i in tqdm(range(num_bootstraping)):

      pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
      acc_test_boot = np.mean(predictions[pred_idx] == target[pred_idx])
      test_accuracies.append(acc_test_boot)

  bootstrap_train_mean = np.mean(test_accuracies)
  return bootstrap_train_mean, (np.percentile(test_accuracies, 2.5), np.percentile(test_accuracies, 97.5))