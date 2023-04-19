import argparse
import gc
import itertools

import numpy as np
from torch.utils.data import DataLoader
from gan.models import *
from gan.datasets import *
from gan.utils import *
import torch
import os
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="spectrum", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=1, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=120, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=3, help="number of residual blocks in generator")
parser.add_argument("--lambda_adv", type=float, default=0.1, help="adv loss weight")
parser.add_argument("--lambda_cyc", type=float, default=1, help="cycle loss weight")

opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("gan/train_out/", exist_ok=True)
os.makedirs("gan/models/", exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_perception = torch.nn.MSELoss()
criterion_D = torch.nn.MSELoss()

cuda = torch.cuda.is_available()

# Initialize generator and discriminator
G = GeneratorResNet(True)
D = Discriminator()

if cuda:
    G = G.cuda()
    D = D.cuda()
    criterion_GAN.cuda()
    criterion_perception.cuda()
    criterion_D.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G.load_state_dict(torch.load("gan/models/G_%d.pth" % (opt.epoch)))
    D.load_state_dict(torch.load("gan/models/D_%d.pth" % (opt.epoch)))
else:
    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(
    G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
vgg = Vgg16().type(Tensor)
# Buffers of previously generated samples

# Image transformations
transforms_ = [
    transforms.ToTensor(),
]

# Training data loader
trainSet = DataLoader(
    TrainSet(transforms_=transforms_, unaligned=True),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)
validationSet = DataLoader(
    ValidationSet(transforms_=transforms_, unaligned=True),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

# ----------
#  Training
# ----------
valid = Variable(Tensor(np.ones((opt.batch_size, *D.output_shape))), requires_grad=False)
fake = Variable(Tensor(np.zeros((opt.batch_size, *D.output_shape))), requires_grad=False)

def printLoss(epoch):

    it = iter(validationSet)
    outPLoss=0
    outDLoss=0
    outRealLoss=0
    outFakeLoss=0
    G.eval()
    D.eval()
    for i in range(100):
        batch=next(it)
        signal = Variable(batch["S"].type(Tensor))
        noisy = Variable(batch["N"].type(Tensor))

        In = nn.InstanceNorm2d(num_features=1)

        fakeSignal = G(noisy) * noisy
        signal = In(signal)
        fakeSignal = In(fakeSignal)

        signalFeatures = vgg(signal.repeat(1, 3, 1, 1))
        fakeFeatures = vgg(fakeSignal.repeat(1, 3, 1, 1))
        perceptualLoss = criterion_perception(signalFeatures[2], fakeFeatures[2])
        dLoss = criterion_GAN(D(fakeSignal), valid)

        loss_real = criterion_D(D(signal), valid)
        loss_fake = criterion_D(D(fakeSignal), fake)

        outPLoss+=perceptualLoss.item()
        outDLoss+=dLoss.item()
        outRealLoss+=loss_real.item()
        outFakeLoss+=loss_fake.item()
    G.train()
    D.train()
    outPLoss/=100
    outDLoss/=100
    outRealLoss/=100
    outFakeLoss/=100
    s="\r[Epoch%d] [D loss: D:%f,real:%f,fake:%f] [G:%f,pLoss:%f,dLoss:%f]"%(epoch,outRealLoss+outFakeLoss,outRealLoss,outFakeLoss,
                                                                              outPLoss+outDLoss,outPLoss,outDLoss)
    sys.stdout.write(s)
    file = open('gan/train_out/info.txt',mode='a+')
    file.write(s)
    file.write('\n')
    file.close()


def run():
    for epoch in range(opt.epoch, opt.n_epochs):
        torch.cuda.empty_cache()

        for i, batch in enumerate(trainSet):
            signal = Variable(batch["S"].type(Tensor))
            noisy = Variable(batch["N"].type(Tensor))

            In = nn.InstanceNorm2d(num_features=1)

            # train G
            optimizer_G.zero_grad()
            fakeSignal = G(noisy)*noisy
            signal = In(signal)
            fakeSignal = In(fakeSignal)

            signalFeatures = vgg(signal.repeat(1, 3, 1, 1))
            fakeFeatures = vgg(fakeSignal.repeat(1, 3, 1, 1))
            perceptualLoss=criterion_perception(signalFeatures[2],fakeFeatures[2])
            dLoss=criterion_GAN(D(fakeSignal),valid)

            loss_G = perceptualLoss + dLoss
            loss_G.backward()
            optimizer_G.step()
            # end

            # train D
            optimizer_D.zero_grad()
            loss_real = criterion_D(D(signal), valid)
            loss_fake = criterion_D(D(fakeSignal.detach()), fake)

            loss_D = (loss_real+loss_fake)/2
            loss_D.backward()
            optimizer_D.step()
            # end

        printLoss(epoch)
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G.state_dict(), "gan/models/G_%d.pth" % (epoch))
            torch.save(D.state_dict(), "gan/models/D_%d.pth" % (epoch))
