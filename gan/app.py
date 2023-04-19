import matplotlib
import numpy as np
from gan.models import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.signal as signal
import torch
import os
import h5py
import obspy

fileDir = r'gan/test_data/'
modelDir = r'gan/models/'
outputDir = "gan/results/"
imageDir = 'gan/results/images/'
os.makedirs(outputDir, exist_ok=True)

transforms_ = [
    transforms.ToTensor(),
]
# load model
G_AB = GeneratorResNet(True)
G_AB.load_state_dict(torch.load(modelDir+"G.pth",map_location='cpu'))
G_AB.eval()

denoiser = G_AB

def processData(signalName,seismicSignal,plot):
    noisySignal = seismicSignal
    _, _, seismicSpectrum = signal.stft(seismicSignal, fs=100, nperseg=30, nfft=60, boundary='zeros')
    _, _, noisySpectrum = signal.stft(noisySignal, fs=100, nperseg=30, nfft=60, boundary='zeros')
    noisyNormalSpec = abs(noisySpectrum)

    noisyNormalSpec = noisyNormalSpec/noisyNormalSpec.max()
    #add channel
    noisyNormalSpec = noisyNormalSpec.reshape(1,1,31,401)
    noisyNormalSpec = torch.Tensor(noisyNormalSpec)
    afterDenoisingSpectrum = denoiser(noisyNormalSpec)

    arr = np.ndarray((31,401), dtype=np.complex128)
    mask = afterDenoisingSpectrum.data.cpu().numpy()[0][0]
    for row in range(31):
        for col in range(401):
            arr[row][col] = complex(noisySpectrum[row][col].real*mask[row][col],noisySpectrum[row][col].imag*mask[row][col])
    resultSpectrum = abs(arr)
    t, afterDenoisingSignal = signal.istft(arr, fs=100, nperseg=30, nfft=60, boundary='zeros')
    if plot == True or plot=='true' or plot=='True' or plot=='TRUE':
        drawImageForCycle(signalName,abs(seismicSpectrum),abs(noisySpectrum),resultSpectrum,seismicSignal,noisySignal,afterDenoisingSignal)
    print(signalName + ' finished')
    return afterDenoisingSignal

def drawImageForCycle(signalName,originalSpectrum,noisySpectrum,resultSpectrum,seismicSignal,noisySignal,afterDenoisingSignal):
    fig = plt.figure(figsize=(30, 15))
    fig.subplots_adjust(wspace=0.15,hspace=0.2)
    fontSize=25
    norm = matplotlib.colors.Normalize(vmin=0, vmax=noisySpectrum.max())
    # left rows
    plt.subplot(3, 2, 1)
    ax = plt.gca()
    plt.text(x=0.05, y=0.95, s='(A)', fontsize=20, verticalalignment="top", horizontalalignment="left",transform = ax.transAxes, color='yellow')
    plt.xticks([0,69,135,201,267,333,401],['0','10','20','30','40','50','60'],fontsize=fontSize)
    plt.yticks([0, 12.4, 24.8], ['0', '20','40'],fontsize=fontSize)
    plt.pcolormesh(originalSpectrum, norm=norm)

    plt.subplot(3, 2, 3)
    ax = plt.gca()
    plt.text(x=0.05, y=0.95, s='(B)', fontsize=20, verticalalignment="top", horizontalalignment="left",transform = ax.transAxes, color='yellow')
    plt.xticks([0, 66, 132, 198, 264, 330, 401], ['0', '10', '20', '30', '40', '50', '60'], fontsize=fontSize)
    plt.yticks([0, 12.4, 24.8], ['0', '20', '40'], fontsize=fontSize)
    noiseSpec = noisySpectrum - resultSpectrum
    plt.pcolormesh(noiseSpec, norm=norm)
    plt.ylabel("Frequency(HZ)", fontsize=fontSize)

    plt.subplot(3, 2, 5)
    ax = plt.gca()
    plt.text(x=0.05, y=0.95, s='(C)', fontsize=20, verticalalignment="top", horizontalalignment="left",transform = ax.transAxes, color='yellow')
    plt.xticks([0, 69, 135, 201, 267, 333, 401], ['0', '10', '20', '30', '40', '50', '60'], fontsize=fontSize)
    plt.yticks([0, 12.4, 24.8], ['0', '20', '40'], fontsize=fontSize)
    plt.pcolormesh(resultSpectrum, norm=norm)
    plt.xlabel("Time(s)", fontsize=fontSize)

    # right rows
    vmin = min(seismicSignal.min(),noisySignal.min(),afterDenoisingSignal.min())
    vmax = max(seismicSignal.max(),noisySignal.max(),afterDenoisingSignal.max())
    plt.subplot(3, 2, 2)
    ax = plt.gca()
    l,=plt.plot(range(6000),seismicSignal, color="black", linewidth=1)
    plt.text(x=0.05, y=0.95, s='(a)', fontsize=20, verticalalignment="top", horizontalalignment="left",transform = ax.transAxes)
    plt.legend(handles=[l], labels=['Original signal'], loc='upper right', fontsize=16)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], ['0', '10', '20', '30', '40', '50', '60'],fontsize=fontSize)
    plt.yticks([vmin, 0, vmax], ['-10', '0', '10'],fontsize=fontSize)

    plt.subplot(3, 2, 4)
    ax = plt.gca()
    noiseSignal = noisySignal - afterDenoisingSignal
    l,=plt.plot(range(6000),noiseSignal, color="black", linewidth=1)
    plt.text(x=0.05, y=0.95, s='(b)', fontsize=20, verticalalignment="top", horizontalalignment="left",transform=ax.transAxes)
    plt.legend(handles=[l], labels=['Noise'], loc='upper right', fontsize=16)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], ['0', '10', '20', '30', '40', '50', '60'],fontsize=fontSize)
    plt.yticks([vmin, 0, vmax], ['-10', '0', '10'],fontsize=fontSize)
    plt.ylabel("Amplitude", fontsize=fontSize)

    plt.subplot(3, 2, 6)
    ax = plt.gca()
    l,=plt.plot(range(6000),afterDenoisingSignal, color="black", linewidth=1)
    plt.text(x=0.05, y=0.95, s='(c)', fontsize=20, verticalalignment="top", horizontalalignment="left",transform=ax.transAxes)
    plt.legend(handles=[l], labels=['Denoised signal'], loc='upper right', fontsize=16)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], ['0', '10', '20', '30', '40', '50', '60'],fontsize=fontSize)
    plt.yticks([vmin, 0, vmax], ['-10', '0', '10'],fontsize=fontSize)
    plt.xlabel("Time(s)", fontsize=fontSize)

    plt.savefig('%s%s.png'%(imageDir,signalName),bbox_inches='tight')
    plt.close()

def doH5(fileName,plot=False):
    dataFile = h5py.File('%s'%(fileName), 'r')
    res = []
    # process data
    for item in  dataFile.keys():
        temp = processData(item, dataFile[item][:], plot)
        res.append(temp)
    # save result
    result = h5py.File('%s'%(resDir(fileName)), "w")
    i = 0
    for item in dataFile.keys():
        result.create_dataset(name=item, data=res[i])
        i += 1
    result.close()
def mseed(fileName, plot=False):
    st = obspy.read(fileName)
    for i in range(len(st)):
        tr = st[i]
        if(len(tr.data)!=6000):
            print("data must have 6000 points,but %s trace %d have %d points. We will just deal with first 6000 points."%(fileName,i,len(tr.data)))
        picName = fileName[fileName.rfind('/')+1:]
        tr.data = processData(picName+'.trace'+str(i), tr.data[:6000], plot);
    st.write(resDir(fileName))
def doDenoise(fileName, plot=False):
    if(fileName.endswith(".mseed")):
        mseed(fileName, plot)
    elif(fileName.endswith(".h5")):
        doH5(fileName, plot)
def dfs(dir, plot=False):
    if (os.path.exists(resDir(dir)) == False):
        os.makedirs(resDir(dir))
    list = os.listdir(dir)
    for item in list:
        item = dir + '/' + item
        if(os.path.isfile(item)):
            doDenoise(item, plot)
        elif(os.path.isdir(item)):
            dfs(item, plot)
def resDir(path):
    return path.replace(fileDir, outputDir,1)
def run(plot=False):
    plot = (plot=='True' or plot=='TRUE' or plot=='true' or plot==True)
    if(plot==True and os.path.exists(imageDir)==False):
        os.makedirs(imageDir)
    print('begin')
    dfs(fileDir, plot)