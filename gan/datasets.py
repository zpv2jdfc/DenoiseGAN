import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
import scipy.signal as signal
import random

noiseFile=r'gan/data/noise.h5'
signalFile=r'gan/data/signal.h5'
validationSignal=r'gan/data/validationSignal.h5'
validationNoise=r'gan/data/validationNoise.h5'

class ValidationSet(Dataset):
    def __init__(self, transforms_=None, unaligned=False, mode="valid"):
        self.transform = transforms.Compose(transforms_)

        dataN_file = h5py.File(validationNoise, 'r')
        dataS_file = h5py.File(validationSignal, 'r')
        self.len_fileN = len(dataN_file.keys())
        self.len_fileS = len(dataS_file.keys())

        self.listN = []
        self.listS = []
        for item in dataN_file.keys():
            self.listN.append(dataN_file[item][:])
        dataN_file.close()
        for item in dataS_file.keys():
            self.listS.append(dataS_file[item][:])
        dataS_file.close()

    def __getitem__(self, index):
        indexS = index
        indexN = random.randint(0, self.len_fileN - 1)

        dataN = self.listN[indexN]
        dataS = self.listS[indexS]

        dataN = dataN + dataS

        _, _, specN = signal.stft(dataN, fs=100, nperseg=30, nfft=60, boundary='zeros')
        _, _, specS = signal.stft(dataS, fs=100, nperseg=30, nfft=60, boundary='zeros')

        specN = abs(specN)
        specS = abs(specS)

        if specN.max()>0:
            specN=specN/specN.max()
            specS=specS/specN.max()

        item_N = self.transform(specN)
        item_S = self.transform(specS)

        return {"N": item_N, "S": item_S}

    def __len__(self):
        return self.len_fileS


class TrainSet(Dataset):
    def __init__(self, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)

        dataN_file = h5py.File(noiseFile, 'r')
        dataS_file = h5py.File(signalFile, 'r')
        self.len_fileN = len(dataN_file.keys())
        self.len_fileS = len(dataS_file.keys())
        self.listN = []
        self.listS = []
        for item in dataN_file.keys():
            self.listN.append(dataN_file[item][:])
        dataN_file.close()
        for item in dataS_file.keys():
            self.listS.append(dataS_file[item][:])
        dataS_file.close()

    def __getitem__(self, index):
        indexS = index
        indexN = random.randint(0, self.len_fileN-1)

        dataN = self.listN[indexN]
        dataS = self.listS[indexS]

        dataN = dataN + dataS

        _, _, specN = signal.stft(dataN, fs=100, nperseg=30,nfft=60,boundary='zeros')  # f:采样频率数组；t:段时间数组；Zxx:STFT结果
        _, _, specS = signal.stft(dataS, fs=100, nperseg=30,nfft=60,boundary='zeros')#31 401

        specN = abs(specN)
        specS = abs(specS)

        if specN.max()>0:
            specN=specN/specN.max()
            specS=specS/specN.max()

        item_N = self.transform(specN)
        item_S = self.transform(specS)

        return {"N":item_N, "S": item_S}
    def __len__(self):
        return self.len_fileS-1
