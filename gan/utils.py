import math
import random
import time
import datetime
import sys

import scipy
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import plot_trigger
from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class Tools:
    def snr(self,signal,pArrival,sArrival=0,dit=300):
        pArrival = int(pArrival)
        sArrival = int(sArrival)
        if dit > pArrival:
            dit = pArrival
        a = np.std(signal[pArrival:pArrival+dit])
        b = np.std(signal[pArrival - dit:pArrival])
        b = max(b, 1e-4)
        if a < 1e-2:
            return 0
        res = 10 * np.log10(a/b)
        if res > 99:
            return 99
        return res


    def correlation(self,x,y):
        EX = x.mean()
        EY = y.mean()
        EXY = (x*y).mean()
        CovXY = EXY - EX*EY
        DX = (x*x).mean() - EX*EX
        DY = (y*y).mean() - EY*EY
        if DX == 0 and DY == 0:
            return 1
        if DX == 0 or DY == 0:
            return 0
        rXY = CovXY/((DX**0.5)*(DY**0.5))
        return rXY
    def pickP(self,signal,k=5):
        nsta=50
        nlta=500
        signal=abs(signal)
        for i in range(500,6000):
            sta=sum(signal[i:i+nsta])/nsta
            lta=sum(signal[i-nlta:i])/nlta
            if sta/lta > k:
                return i
        return -1
