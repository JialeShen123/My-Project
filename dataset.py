import pickle
from os.path import join
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.contrib.concurrent import process_map
pd.options.mode.chained_assignment = None


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-5)


def zmean(x):
    return x - x.mean()


class Fluxnet(Dataset):
    """Fluxnet dataset."""

    def __init__(self, root_dir, site_csv_names, num_workers=16, is_train=True, transform=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.num_workers = num_workers
        results = process_map(self.process_file, site_csv_names, max_workers=self.num_workers, chunksize=1)

        self.samples = []
        self.ground_truth = []
        self.sparse = []
        self.sparse_max = []
        self.sparse_min = []
        self.le_all = []
        self.site_names = []
        self.site_dates = []
        for result in results:
            self.samples.extend(result['samples'])
            self.ground_truth.extend(result['ground_truth'])
            self.sparse.extend(result['sparse'])
            self.sparse_max.extend(result['sparse_max'])
            self.sparse_min.extend(result['sparse_min'])
            self.le_all.extend(result['le_all'])
            self.site_names.extend(result['site_names'])
            self.site_dates.extend(result['site_dates'])

    def process_file(self, line):
        result = {
            'samples': [],
            'ground_truth': [],
            'sparse': [],
            'sparse_max': [],
            'sparse_min': [],
            'le_all': [], # 直接从文件中读取的连续数据值;
            'site_names': [],
            'site_dates': []
        }

        line = line.strip()
        csv_path = join(self.root_dir, line)
        site = pd.read_csv(csv_path)
        dates = sorted(site['Date'].unique())
        for date in dates:
            site_date = site[site['Date'] == date]
            inputs = site_date[['VPD_F', 'TA_F', 'PA_F',
                                'WS_F', 'P_F', 'LW_IN_F', 'SW_IN_F']]
            inputs = self.gaussify_input(inputs).to_numpy()
            inputs = inputs[[3, 9, 15, 21, 27, 33, 39, 45], :]
            le_all = site_date["LE_F_MDS"].to_numpy().astype(np.float32)
            # choice A: use only one LE observation per day
            # TERRA,SPOT Overpass Time ~10:30
            sparse = le_all[[21,21,21,21]]

            #  FY-3D Satellite Overpass Time ~14:00
            # sparse = le_all[[28, 28, 28, 28]]

            max_LE = np.asarray([500.],dtype=np.float32)
            min_LE = np.asarray([-50.],dtype=np.float32)

            # choice B: use 4 LE observations per day
            # sparse = le_all[[3, 21, 27, 45]]
            # max_LE = np.asarray([np.max(le_all[[3,21,27,45]])])
            # min_LE = np.asarray([np.min(le_all[[3,21,27,45]])])

            result['le_all'].append(le_all)
            result['samples'].append(inputs.astype(np.float32))
            result['sparse_max'].append(max_LE)
            result['sparse_min'].append(min_LE)
            result['site_names'].append(line.split('.')[0])
            result['site_dates'].append(date)
            result['sparse'].append(sparse.astype(np.float32))
            target_array = site_date["LE_F_MDS"].to_numpy()
            target = target_array.mean().astype(np.float32)
            result['ground_truth'].append(target)

        return result
    # clip防止异常值
    def normalize_input(self, inputs):
        inputs.loc[:, "VPD_F"] = (
            np.clip(inputs["VPD_F"].to_numpy(), 0, 100))/100
        inputs.loc[:, "WS_F"] = (
            np.clip(inputs["WS_F"].to_numpy(), 0, 100))/100
        inputs.loc[:, "P_F"] = (np.clip(inputs["P_F"].to_numpy(), 0, 10))/10
        inputs.loc[:, "LW_IN_F"] = (
            np.clip(inputs["LW_IN_F"].to_numpy(), 0, 1000))/1000
        inputs.loc[:, "SW_IN_F"] = (
            np.clip(inputs["SW_IN_F"].to_numpy(), 0, 2000))/2000
        inputs.loc[:, "TA_F"] = (
            np.clip(inputs["TA_F"].to_numpy(), -70, 70) + 70)/140
        inputs.loc[:, "PA_F"] = (
            np.clip(inputs["PA_F"].to_numpy(), 0, 102))/102
        return inputs

    def gaussify_input(self, inputs):
        # normalize inputs by its mean and std
        inputs.loc[:, "VPD_F"] = zscore(inputs["VPD_F"].to_numpy())
        inputs.loc[:, "WS_F"] = zscore(inputs["WS_F"].to_numpy())
        inputs.loc[:, "P_F"] = zscore(inputs["P_F"].to_numpy())
        inputs.loc[:, "LW_IN_F"] = zscore(inputs["LW_IN_F"].to_numpy())
        inputs.loc[:, "SW_IN_F"] = zscore(inputs["SW_IN_F"].to_numpy())
        inputs.loc[:, "TA_F"] = zscore(inputs["TA_F"].to_numpy())
        inputs.loc[:, "PA_F"] = zscore(inputs["PA_F"].to_numpy())
        return inputs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {}
        sample['inputs'] = self.samples[idx]
        sample['target'] = np.expand_dims(
            np.asarray(self.ground_truth[idx]), 0)
        sample['sparse'] = self.sparse[idx]
        sample['sparse_max'] = self.sparse_max[idx]
        sample['sparse_min'] = self.sparse_min[idx]
        sample["le_all"] = self.le_all[idx]
        if not self.is_train:
            sample["site_name"] = self.site_names[idx]
            sample["site_date"] = self.site_dates[idx]
        return sample
