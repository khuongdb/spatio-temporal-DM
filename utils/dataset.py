from torch.utils import data
import pandas as pd
import numpy as np
import torch
import os


subject = 5

class Data_preprocess_starmen:
    def __init__(self, info_path):
        self.p = 0.2
        self.info_path = info_path
        self.catalog = pd.read_csv(self.info_path)
        self.generate_init_data()

    def generate_init_data(self):
        self.catalog = self.catalog.loc[:, self.catalog.columns != 'id']
        tau_list = self.catalog.iloc[:, 0]
        alpha_list = self.catalog.iloc[:, 1]
        age_list = self.catalog.iloc[:, 2]

        npy_path_list = self.catalog.iloc[:, 3]
        # change relative path
        relative_path = self.info_path.split("df.csv")[0]
        npy_path_list = npy_path_list.apply(lambda x: os.path.join(relative_path, "images", x.split("/images/")[-1]))

        first_age_list = pd.DataFrame(data=[age_list[int(i / 10) * 10] for i in range(10 * subject)], columns=['first_age'])
        subject_list = pd.DataFrame(data=[int(i / 10) for i in range(10 * subject)], columns=['subject'])
        timepoint_list = pd.DataFrame(data=[i % 10 for i in range(10 * subject)], columns=['timepoint'])

        self.catalog = pd.concat(
            [npy_path_list, subject_list, tau_list, age_list, timepoint_list, first_age_list, alpha_list], axis=1)
        self.catalog = self.catalog.rename(columns={'t': 'age', 'tau': 'baseline_age'})





    def correct_img_path(old_path, new_path):
        return os.path.join(new_path, old_path.split("/images/")[-1])

    def generate_train_test(self, fold):
        test_num = int(subject * self.p)

        test_index = np.arange(test_num, dtype=int) + test_num * fold
        train_index = np.setdiff1d(np.arange(subject, dtype=int), test_index)

        train = self.catalog.loc[self.catalog['subject'].isin(train_index)]
        train_data = train.set_index(pd.Series(range(int((1 - self.p) * 10 * subject))))

        test = self.catalog.loc[self.catalog['subject'].isin(test_index)]
        test_data = test.set_index(pd.Series(range(int(self.p * 10 * subject))))

        return train_data, test_data

    def generate_all(self):
        return self.catalog


    def generate_XY(self, train_data):
        N = len(train_data.index)
        I = N // 10

        delta_age = train_data['age'] - train_data['baseline_age']
        ones = pd.DataFrame(np.ones(shape=[N, 1]))
        X = pd.concat([ones, delta_age, train_data['baseline_age']], axis=1)

        zero = pd.DataFrame(np.zeros(shape=[10, 2]))
        for i in range(I):
            y = X.iloc[i * 10:(i + 1) * 10, :2]
            y = y.set_axis([0, 1], axis=1)
            if i == 0:
                zeros = pd.concat([zero for j in range(I - 1)], axis=0)
                Y = pd.concat([y, zeros], axis=0).reset_index(drop=True)
            elif i != I - 1:
                zeros1 = pd.concat([zero for j in range(i)], axis=0)
                zeros2 = pd.concat([zero for j in range(I - 1 - i)], axis=0).reset_index(drop=True)
                yy = pd.concat([zeros1, y, zeros2], axis=0).reset_index(drop=True)
                Y = pd.concat([Y, yy], axis=1)
            else:
                zeros = pd.concat([zero for j in range(I - 1)], axis=0)
                yy = pd.concat([zeros, y], axis=0).reset_index(drop=True)
                Y = pd.concat([Y, yy], axis=1)

        X = torch.tensor(X.values)
        Y = torch.tensor(Y.values)
        return X, Y


class Dataset_starmen(data.Dataset):
    def __init__(self, image_path, subject, baseline_age, age, timepoint, first_age, alpha):
        self.image_path = image_path
        self.subject = subject
        self.baseline_age = baseline_age
        self.age = age
        self.timepoint = timepoint
        self.first_age = first_age
        self.alpha = alpha

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.image_path[index]
        y = self.subject[index]
        z = self.baseline_age[index]
        u = self.age[index]
        v = self.timepoint[index]
        w = self.first_age[index]
        a = self.alpha[index]
        return x, y, z, u, v, w, a