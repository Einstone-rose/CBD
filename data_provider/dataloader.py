import os
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader
from scipy.io import arff
import pickle

class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode=None):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        
        x_train = dataset['samples']
        y_train = dataset['labels']

        if len(x_train.shape) < 3:
            x_train = x_train.unsqueeze(2)
        if x_train.shape.index(min(x_train.shape)) != 1:  # make sure the Channels in second dim
            x_train = x_train.permute(0, 2, 1)  # [batch size, nvars, seq_len]

        if isinstance(x_train, np.ndarray):
            self.x_data = torch.from_numpy(x_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = x_train
            self.y_data = y_train
        self.len = x_train.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len
    

def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data

def load_UCR(Path='data/', folder='Cricket'):
    train_path = Path + folder + '/' + folder + '_TRAIN.arff'
    test_path = Path + folder + '/' + folder + '_TEST.arff'
    TRAIN_DATA = []
    TRAIN_LABEL = []
    label_dict = {}
    label_index = 0
    with open(train_path, encoding='UTF-8', errors='ignore') as f:
        data, meta = arff.loadarff(f)
        f.close()
    if type(data[0][0]) == np.ndarray:  # multivariate
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            raw_data_list = raw_data.tolist()
            # print(raw_data_list)
            TRAIN_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = data[index][0]
            raw_label = data[index][1]
            TEST_LABEL.append(label_dict[raw_label])
            raw_data_list = raw_data.tolist()
            TEST_DATA.append(np.array(raw_data_list).astype(np.float32).transpose(-1, 0))

        index = np.arange(0, len(TRAIN_DATA))
        np.random.shuffle(index)

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TRAIN_DATA)[index], np.array(TRAIN_LABEL)[index]], \
               [np.array(TEST_DATA), np.array(TEST_LABEL)]

    else:  # univariate
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            if label_dict.__contains__(raw_label):
                TRAIN_LABEL.append(label_dict[raw_label])
            else:
                label_dict[raw_label] = label_index
                TRAIN_LABEL.append(label_index)
                label_index += 1
            TRAIN_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TEST_DATA = []
        TEST_LABEL = []
        with open(test_path, encoding='UTF-8', errors='ignore') as f:
            data, meta = arff.loadarff(f)
            f.close()
        for index in range(data.shape[0]):
            raw_data = np.array(list(data[index]))[:-1]
            raw_label = data[index][-1]
            TEST_LABEL.append(label_dict[raw_label])
            TEST_DATA.append(np.array(raw_data).astype(np.float32).reshape(-1, 1))

        TRAIN_DATA = padding_varying_length(np.array(TRAIN_DATA))
        TEST_DATA = padding_varying_length(np.array(TEST_DATA))

        return [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], [
            np.array(TEST_DATA), np.array(TEST_LABEL)]
    

def data_generator(data_path, configs, data_type, training_mode):
    batch_size = configs.batch_size
    common_list = ["sleepEDF", "HAR", "Epilepsy"]
    uea_list = ["ECG5000", "MotorImagery", "FingerMovements", "PhonemeSpectra", "UWaveGestureLibraryAll", "ArticularyWordRecognition", "SelfRegulationSCP1", "SpokenArabicDigits", "FordB"]
#     if "_1p" in training_mode:
#         train_dataset = torch.load(os.path.join(data_path, "folds_data", "train_1per.pt"))
#     elif "_5p" in training_mode:
#         train_dataset = torch.load(os.path.join(data_path, "folds_data", "train_5per.pt"))
#     elif "_10p" in training_mode:
#         train_dataset = torch.load(os.path.join(data_path, "/folds_data/train_10per.pt"))
#     elif "_50p" in training_mode:
#         train_dataset = torch.load(os.path.join(data_path, "/folds_data/train_50per.pt"))
#     elif "_75p" in training_mode:
#         train_dataset = torch.load(os.path.join(data_path, "/folds_data/train_75per.pt"))
#     else:
#         train_dataset = torch.load(os.path.join(data_path, "train.pt"))
#     print(train_dataset["samples"].shape)
#     valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
#     test_dataset = torch.load(os.path.join(data_path, "test.pt"))

#     train_dataset = Load_Dataset(train_dataset, configs, training_mode)
#     valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
#     test_dataset = Load_Dataset(test_dataset, configs, training_mode)
    
    if data_type in common_list:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))
        valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
        test_dataset = torch.load(os.path.join(data_path, "test.pt"))
        train_dataset = Load_Dataset(train_dataset, configs)
        valid_dataset = Load_Dataset(valid_dataset, configs)
        test_dataset = Load_Dataset(test_dataset, configs)
    elif data_type in uea_list:
        _, train_data, test_data = load_UCR(Path="./dataset/", folder=data_type)
        train_data_ts = torch.from_numpy(train_data[0]).float()  # bs, sl, ch
        bs, fl, dm = train_data_ts.shape
        if fl % 2 != 0:
            train_data_ts = torch.cat((train_data_ts, train_data_ts[:, -1:, :].expand(bs, 1, dm)), dim=1)
        train_data_ts = train_data_ts.transpose(-1, -2)
        train_data_label = torch.from_numpy(train_data[1]).long()
        test_data_ts = torch.from_numpy(test_data[0]).float()
        bs, fl, dm = test_data_ts.shape
        if fl % 2 != 0:
            test_data_ts = torch.cat((test_data_ts, test_data_ts[:, -1:, :].expand(bs, 1, dm)), dim=1)
        test_data_ts = test_data_ts.transpose(-1, -2)
        test_data_label = torch.from_numpy(test_data[1]).long()
        train_dataset = TensorDataset(train_data_ts, train_data_label)
        test_dataset = TensorDataset(test_data_ts, test_data_label)
        # if "finetune" or "linear_evaluation" in training_mode:
        #     val_size = int(len(train_dataset) * 0.1)
        #     print(f'random split val dataset, val dataset size is {val_size}')
        #     train_size = len(train_dataset) - val_size
        #     train_dataset, valid_dataset = random_split(train_dataset, [train_size, val_size])
        # else:
        #     valid_dataset = train_dataset
        valid_dataset = train_dataset
        
    if train_dataset.__len__() < batch_size:
        batch_size = train_dataset.__len__()
    if test_dataset.__len__() < batch_size:
        ts_batch_size = test_dataset.__len__()
    else: 
        ts_batch_size = batch_size
    if valid_dataset.__len__() < batch_size:
        val_batch_size = valid_dataset.__len__()
    else: 
        val_batch_size = batch_size

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=val_batch_size, shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=ts_batch_size, shuffle=False, drop_last=configs.drop_last, num_workers=0)

    return train_loader, valid_loader, test_loader
