import time
import numpy as np
import torch
import torch.nn as nn
from utils import MSE, sMAPE, np_sMAPE
import pandas as pd


class ResidualDRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, config, ohe_headers):
        super(ResidualDRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        self.ohe_headers = ohe_headers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        # self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=config['lr_anneal_step'],
                                                         gamma=config['lr_anneal_rate'])
        
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        for e in range(self.max_epochs):
            self.scheduler.step()
            epoch_loss = self.train()
            epoch_val_loss = self.val()
            print('Validation Loss: %f' % epoch_val_loss)
        print('Total Training Mins: %5.2f' % ((time.time()-start_time)/60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (train, resid, trend, seasonal, mean, std, val, info_cat) in enumerate(self.dl):
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(train, resid, trend, seasonal, mean, std, val, info_cat)
            epoch_loss += loss
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # LOG EPOCH LEVEL INFORMATION
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (self.epochs, self.max_epochs, epoch_loss))

        return epoch_loss

    def train_batch(self, train, resid, trend, seasonal, mean, std, val, info_cat):
        self.optimizer.zero_grad()
        network_pred, network_act, _, _, _ = self.model(train, resid, trend, seasonal, mean, std, val, info_cat)

        loss = MSE(network_pred, network_act)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        return float(loss)

    def val(self):
        self.model.eval()
        with torch.no_grad():
            hold_out_loss = 0
            for batch_num, (train, resid, trend, seasonal, mean, std, val, info_cat) in enumerate(self.dl):
                _, _,_, hold_out_pred, hold_out_act= self.model(train, resid, trend, seasonal, mean, std, val, info_cat)
                hold_out_loss += sMAPE(hold_out_pred, hold_out_act)
            hold_out_loss = hold_out_loss / (batch_num + 1)
        return float(hold_out_loss)


