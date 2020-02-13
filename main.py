"""*********************************************************************************************

                       Residual Dilated RNNs with STL Decomposition

*********************************************************************************************"""
#%% Importing Libraries

import pandas as pd
from torch.utils.data import DataLoader
from data_loading import create_val_dataset, create_dataset, SeriesDataset
from config import get_config
from model import ResidualDRNN_Main
from trainer import ResidualDRNNTrainer

#%%

print('loading config')
config = get_config('Monthly')
print('loading data')
info = pd.read_csv('../data/M4-info.csv')

train_path = '../data/Train/%s-train.csv' % (config['variable'])
test_path = '../data/Test/%s-test.csv' % (config['variable'])

mode='Testing'

if mode =='Training':
    train, val = create_val_dataset(train_path, config['output_size'])
    # Returning both train, and a decomposed version of train using STL
    dataset = SeriesDataset(train, val, info, config['variable'], config['seasonality'], config['chop_val'], config['device'])
elif mode=='Testing':
    train, test = create_dataset(train_path, test_path)
    # Returning both train, and a decomposed version of train using STL
    dataset = SeriesDataset(train, test, info, config['variable'], config['seasonality'], config['chop_val'], config['device'])


dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

#%%

model = ResidualDRNN_Main(config=config)
tr = ResidualDRNNTrainer(model, dataloader, config, ohe_headers=dataset.dataInfoCatHeaders)
tr.train_epochs()


