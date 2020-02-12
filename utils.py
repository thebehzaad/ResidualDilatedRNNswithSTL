import torch
import torch.nn as nn
import numpy as np

def MSE(predictions, actuals):
    predictions = predictions.float()
    actuals = actuals.float()
    sumf=torch.sum((predictions - actuals)**2)
    return (sumf/predictions.numel())

def sMAPE(predictions, actuals):
    predictions = predictions.float()
    actuals = actuals.float()
    sumf = torch.sum(torch.abs(predictions - actuals) / (torch.abs(predictions) + torch.abs(actuals)))
    return ((2 * sumf) / predictions.numel()) * 100

def np_sMAPE(predictions, actuals):
    predictions = torch.from_numpy(np.array(predictions))
    actuals = torch.from_numpy(np.array(actuals))
    return float(sMAPE(predictions, actuals))

if __name__ == '__main__':
    test1 = torch.rand(10,4,3)
    test2 = torch.rand(10,4,3)
    vec_loss = sMAPE(test1, test2)




