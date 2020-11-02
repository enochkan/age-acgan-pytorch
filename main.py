"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os, sys
import numpy as np
import random, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc, dice_loss
from network import _netG, _netD
from folder import ImageFolder
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import cv2
torch.autograd.set_detect_anomaly(True)
np.set_printoptions(threshold=sys.maxsize)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height/width of the input image to network')
parser.add_argument('--imageHeight', type=int, default=172, help='the height of the input image to network')
parser.add_argument('--imageWidth', type=int, default=344, help='the width of the input image to network')
parser.add_argument('--nz', type=int, default=102, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--glr', type=float, default=0.0002, help='generator learning rate, default=0.0002')
parser.add_argument('--dlr', type=float, default=0.0002, help='discriminator learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--test', default=False, help='Testing')
parser.add_argument('--gen_samples', type=int, default=20, help='Number of test samples to generate')
parser.add_argument('--save_from', type=int, default=1500, help='limiting models saved')
parser.add_argument('--process_mask', default=True, help='post process the mask')

opt = parser.parse_args()
print(opt)

# specify the gpu id if using only 1 gpu
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
if opt.ngpu > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print('Available devices ', torch.cuda.device_count())
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# datase t
if opt.dataset == 'imagenet':
    # folder dataset
    dataset = ImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Scale(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        classes_idx=(10, 20)
    )
elif opt.dataset != 'imagenet':
    dataset = ImageFolder(root=opt.dataroot, transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.25,)),
        ]))
else:
    raise NotImplementedError("No such dataset {}".format(opt.dataset))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3

# testing script
def test_netG(generator):
    for sample in range(0, opt.gen_samples):
        eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
        eval_noise = Variable(eval_noise)
        eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
        # eval_label = np.random.randint(0, num_classes, opt.batchSize)
        eval_label = np.ones(opt.batchSize, dtype=int)
        eval_onehot = np.zeros((opt.batchSize, opt.num_classes))
        eval_onehot[np.arange(opt.batchSize), eval_label] = 1
        eval_noise_[np.arange(opt.batchSize), :opt.num_classes] = eval_onehot[np.arange(opt.batchSize)]
        eval_noise_ = (torch.from_numpy(eval_noise_))
        eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))
        fake, _ = generator(eval_noise)
        fake = fake.clone()
        if opt.process_mask:
            fake[:,:,:,int(opt.imageWidth/2):int(opt.imageWidth)].data[fake[:,:,:,int(opt.imageWidth/2):int(opt.imageWidth)] >= 0.5] = 1
            fake[:,:,:,int(opt.imageWidth/2):int(opt.imageWidth)].data[fake[:,:,:,int(opt.imageWidth/2):int(opt.imageWidth)] < 0.5] = 0
        vutils.save_image(
            fake.data,
            '%s/test_samples_%03d_%s.png' % (opt.outf, sample, str(eval_label))
        )
        print('generated '+str(sample)+' test image')
    sys.exit()

# Define the generator and initialize the weights
if opt.dataset != 'imagenet':
    if opt.ngpu > 1:
        print('Use {} GPUs'.format(torch.cuda.device_count()))
        netG = _netG(ngpu, nz)
    else:
        netG = _netG(ngpu, nz)
netG.apply(weights_init)
if opt.netG != '':
    if opt.test:
        print('loading model...')
        netG.load_state_dict(torch.load(opt.netG))
        print('loading model successful...')
        test_netG(netG)
    else:
        netG.load_state_dict(torch.load(opt.netG))
print(netG)
# Define the discriminator and initialize the weights
if opt.dataset != 'imagenet':
    if opt.ngpu > 1:
        print('Use {} GPUs'.format(torch.cuda.device_count()))
        netD = _netD(ngpu, num_classes)
    else:
        netD = _netD(ngpu, num_classes)
netD.apply(weights_init)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
#     print(netD)
#     sys.exit()
print(netD)

# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.CrossEntropyLoss()

# tensor placeholders
input = torch.FloatTensor(opt.batchSize, 1, opt.imageHeight, opt.imageWidth)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.batchSize)
aux_label = torch.LongTensor(opt.batchSize)
real_label = 1
fake_label = 0

# if using cuda
if opt.cuda and ngpu > 1:
    netD.cuda(1)
    netG.cuda(1)
    dis_criterion.cuda(1)
    aux_criterion.cuda(1)
    input, dis_label, aux_label = input.cuda(1), dis_label.cuda(1), aux_label.cuda(1)
    noise, eval_noise = noise.cuda(1), eval_noise.cuda(1)
else:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
eval_label = np.random.randint(0, num_classes, opt.batchSize)
eval_onehot = np.zeros((opt.batchSize, num_classes))
eval_onehot[np.arange(opt.batchSize), eval_label] = 1
eval_noise_[np.arange(opt.batchSize), :num_classes] = eval_onehot[np.arange(opt.batchSize)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.dlr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.glr, betas=(opt.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
start_time = time.time()
a = ''; i_ = 0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_(real_cpu.size()).copy_(real_cpu)
        dis_label.resize_(batch_size).fill_(real_label)
        aux_label.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(input)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        aux_label.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        dis_label.data.fill_(fake_label)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.data.fill_(real_label)  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        errG = dis_errG + aux_errG
        errG.backward()
        D_G_z2 = dis_output.data.mean()
        optimizerG.step()

        # compute the average loss
        curr_iter = epoch * len(dataloader) + i
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter
        all_loss_G += errG.data
        all_loss_D += errD.data
        all_loss_A += accuracy
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)

        #save loss
        a+=('('+str(i_)+','+str(errG.data.item())+')')
        with open('./logs/loss_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([a])

        print('[%d/%d][%d/%d] time: %4.4f, Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Dice: %.4f Acc: %.4f (%.4f)'
              % (epoch, opt.niter, i, len(dataloader), time.time() - start_time,
                 errD.data, avg_loss_D, errG.data, avg_loss_G, D_x, D_G_z1, D_G_z2, 1, accuracy, avg_loss_A))
        if i % len(dataloader) == 0:
            vutils.save_image(
                real_cpu, '%s/real_samples.png' % opt.outf)
            print('Label for eval = {}'.format(eval_label))
            el = '{}'.format(eval_label)
            fake = netG(eval_noise)
            vutils.save_image(
                fake.data,
                '%s/fake_samples_epoch_%03d_%s.png' % (opt.outf, epoch,el)
            )
        i_+=1
    if epoch > opt.save_from:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
