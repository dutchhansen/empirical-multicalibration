import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TemperatureScaling:
    """
    Temperature scaling method for calibration, for binary classification.
    Allows for fixed or optimized temperature.
    """
    def __init__(self, params):
        self.optimized = params['optimized']
        self.t = params['temperature']

        # must fix temperature or optimize it, not both
        assert self.optimized == (self.t is None), \
            'Must provide temperature XOR set optimized to True;' + \
            'if optimized, temperature learned from holdout.'

    def fit(self, logits, labels, subgroups):
        '''
        Parameters:
            :logits: numpy array of logits for positive class
            :labels: numpy array of true labels
        '''
        if self.optimized:
            logits_t = torch.tensor(logits, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
            self.mod = TemperatureModule()
            self.denom = self.mod.set_temperature(logits_t, labels_t)
        else:
            self.denom = self.t

    def batch_predict(self, f_xs, groups):
        f_t = torch.tensor(f_xs, dtype=torch.float32)
        f_sm = F.softmax(f_t / self.denom, dim=1)[:,1]
        # print("f before temp scaling: ", F.softmax(f_t, dim=1)[:,1].numpy()[:5])
        # print("f after temp scaling: ", f_sm.numpy()[:5])
        return f_sm.numpy()


class TemperatureModule(nn.Module):
    '''
    NN Module for finding temperature which minimizes CE.
    Code adapted from Geoff Pleiss:
        https://github.com/gpleiss/temperature_scaling/tree/master.
    '''
    def __init__(self):
        super(TemperatureModule, self).__init__()
        initial = 1.5
        self.temperature = nn.Parameter(torch.ones(1) * initial)
    
    def temperature_scale(self, logits):
        '''
        Scale logits by temperature.
        Parameters:
            :logits: torch tensor of logits
        '''
        t = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / t
    
    def set_temperature(self, logits, labels):
        '''
        Tune the tempearature of the model (using the validation set).
        As in prior works, optimize CE.

        Parameters:
            :logits: tensor of logits for positive class
            :labels: tensor of true labels
        '''
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        # print ce loss before optimization
        # print("CE loss before optimization: ", 
        #       criterion(self.temperature_scale(logits), labels).item())

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        
        # optimize temperature
        optimizer.step(eval)

        # print ce loss after optimization
        # print("CE loss after optimization: ", 
        #       criterion(self.temperature_scale(logits), labels).item())

        # return denomenator
        print("Temperature: ", self.temperature.item())
        return self.temperature.item()
