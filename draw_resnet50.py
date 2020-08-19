from __future__ import print_function
import os, time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm

import models
from filter import *
from scipy.ndimage import filters
from compute_flops import print_model_param_flops
from collections import OrderedDict
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--scratch',default='', type=str,
                    help='the PATH to the pruned model')
# filter
parser.add_argument('--filter', default='none', type=str, choices=['none', 'lowpass', 'highpass'])
parser.add_argument('--sigma', default=1.0, type=float, help='gaussian filter hyper-parameter')

# sparsity
parser.add_argument('--sparsity_gt', default=0, type=float, help='sparsity controller')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--port", type=str, default="15000")

# swa
parser.add_argument('--swa', default=False, help='SWALP start epoch')
parser.add_argument('--swa_start', type=int, default=60, metavar='N',
                    help='SWALP start epoch')

# load pretrained
parser.add_argument('--load-model', default=None, type=str)
parser.add_argument('--val-cacc', action="store_true")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

gpu = args.gpu_ids
gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
    id = int(gpu_id)
    args.gpu_ids.append(id)
print(args.gpu_ids)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])

os.environ['MASTER_PORT'] = args.port
torch.distributed.init_process_group(backend="nccl")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.Lambda(lambda x: filters.gaussian_filter(x, args.sigma) if args.filter == 'lowpass' else x),
                           transforms.Lambda(lambda x: my_gaussian_filter_2(x, 1/args.sigma, args.filter) if args.filter == 'highpass' else x),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: torch.where(x > args.sparsity_gt, x, torch.zeros_like(x)) if args.sparsity_gt > 0 else x),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=16, pin_memory=True)



if args.dataset == 'imagenet':
    model = models.__dict__[args.arch](pretrained=False)
    if args.load_model:
        state_dict = torch.load(args.load_model)['state_dict']
        model_state_dict = model.state_dict()
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        
    if args.swa == True:
        swa_n = 0
        swa_model = models.__dict__[args.arch](pretrained=False)
    if args.scratch:
        checkpoint = torch.load(args.scratch)
        if args.dataset == 'imagenet':
            cfg_input = checkpoint['cfg']
            model = models.__dict__[args.arch](pretrained=False, cfg=cfg_input)
            if args.swa == True:
                swa_model = models.__dict__[args.arch](pretrained=False, cfg=cfg_input)
    if args.cuda:
        model.cuda()
        if args.swa == True:
            swa_model.cuda()
    if len(args.gpu_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids)
        if args.swa == True:
            swa_model = torch.nn.parallel.DistributedDataParallel(swa_model, device_ids=args.gpu_ids)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    if args.cuda:
        model.cuda()
    if len(args.gpu_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids)

if args.dataset == 'imagenet':
    pruned_flops = print_model_param_flops(model, 224)
    
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if args.swa == True:
            swa_model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        
print(model)
state_dict = model.state_dict()
for name in tqdm(state_dict.keys()):
    if 'bn' in name and 'weight' in name:
        weight = state_dict[name]
        length = len(weight)
        plt.ylim(0, 1)
        plt.bar(list(range(length)), abs(weight.cpu().numpy()) / max(abs(weight.cpu().numpy())))
        plt.savefig(name + '.jpg')
        plt.clf()