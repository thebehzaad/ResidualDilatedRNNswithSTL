import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from stldecompose import decompose


def read_file(file_location):
    series = []
    ids = []
    with open(file_location, 'r') as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
    # for i in range(1, len(data)):
        row = data[i].replace('"', '').split(',')
        series.append(np.array([float(j) for j in row[1:] if j != ""]))
        ids.append(row[0])

    series = np.array(series)
    return series


def create_val_dataset(train_file_location, output_size):
    train = read_file(train_file_location)
    val = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        train[i] = train[i][:-output_size]
    return train, np.array(val)


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return np.array(train), train_len_mask


def create_dataset(train_file_location, test_file_location):
    train = read_file(train_file_location)
    test = read_file(test_file_location)
    return train, test


def stl_decomposition(train, variable, seasonality):
    
    train_data=pd.DataFrame(train.transpose())
    index = pd.date_range(start = '20/01/2000', periods=train_data.shape[0], freq=variable[0])
    train_data.set_index(index,inplace=True)
    trend=[]
    seasonal=[]
    resid=[]
    mean_resid=[]
    std_resid=[]
    for i in range(train_data.shape[1]):
        decomp = decompose(train_data[i], period=seasonality)
        trend.append(decomp.trend.values)
        seasonal.append(decomp.seasonal.values)
        mean_resid.append(np.mean(decomp.resid.values))
        std_resid.append(np.std(decomp.resid.values))
        resid.append((decomp.resid.values-np.mean(decomp.resid.values))/np.std(decomp.resid.values))
    
    return np.array(trend), np.array(seasonal), np.array(resid), np.array(mean_resid), np.array(std_resid)


class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, info, variable, seasonality, chop_value, device):
        dataTrain, mask = chop_series(dataTrain, chop_value)        
        dataTrend, dataSeasonal, dataResid, meanResid, stdResid = stl_decomposition(dataTrain, variable, seasonality)
        
        self.dataInfoCatOHE = pd.get_dummies(info[info['SP'] == variable]['category'])
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
        self.dataResid = [torch.tensor(dataResid[i]) for i in range(len(dataResid))]
        self.dataTrend= [torch.tensor(dataTrend[i]) for i in range(len(dataTrend))]
        self.dataSeasonal= [torch.tensor(dataSeasonal[i]) for i in range(len(dataSeasonal))]
        self.meanResid= [torch.tensor(meanResid[i]) for i in range(len(meanResid))]
        self.stdResid= [torch.tensor(stdResid[i]) for i in range(len(stdResid))]
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal)) if mask[i]]
        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device),\
               self.dataResid[idx].to(self.device),\
               self.dataTrend[idx].to(self.device),\
               self.dataSeasonal[idx].to(self.device),\
               self.meanResid[idx].to(self.device),\
               self.stdResid[idx].to(self.device),\
               self.dataVal[idx].to(self.device), \
               self.dataInfoCat[idx].to(self.device)


