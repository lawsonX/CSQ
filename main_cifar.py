from lib2to3.pgen2.grammar import opmap_raw
import os
import math
import datetime
import argparse
import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
from cifar10.network.resnetcs20 import ResNet
from imagenet.networks.resnetcs18 import ResNet18
from imagenet.networks.resnetcs50 import ResNet50
from imagenet.networks.vgg import VGG19bn
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='Training a ResNet on CIFAR-10 with Continuous Sparsification')
# parser.add_argument('--which-gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--data',metavar='DIR',default='/home/datasets/imagenet',help='path to dataset')
parser.add_argument('--batch-size', type=int, default=96, metavar='N', help='input batch size for training/val/test (default: 128)')
parser.add_argument('--epochs', type=int, default=440, help='number of epochs to train (default: 300)')
parser.add_argument('--solidize', type=int, default=430, help='The epoch to solidize the mask')
parser.add_argument('--rewind', type=int, default=200, help='The epoch to rewind the global tempreture')
parser.add_argument('--classes', type=int, default=10, help='class of output')
parser.add_argument('--Nbits', type=int, default=6, help='quantization bitwidth for weight')
parser.add_argument('--target', type=int, default=4, help='Target Nbit')
parser.add_argument('--act', type=int, default=0, help='quantization bitwidth for activation')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--lmbda', type=float, default=0.001, help='lambda for L1 mask regularization (default: 1e-8)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--save_file', type=str, default='TIM_res18_A0N8T7_Steplr00001_96_e200', help='save path of weight and log files')
parser.add_argument('--log_file', type=str, default='train.log', help='save path of weight and log files')
parser.add_argument('-a','--arch', default='ResNet', help= 'ResNet for resnet20 on cifar10, ResNet18,VGG19bn for imagenet&TinyImagenet')
parser.add_argument('--warmup',dest='warmup',action='store_true',help='warmup learning rate for the first 5 epochs')

args = parser.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def get_ratio_one(model):
    mask = [m.mask for m in model.mask_modules]
    total_ele = 0
    ones = 0
    for iter in range(len(mask)):
        t = mask[iter].numel()
        o = (mask[iter] >= 0.5).sum().item()
        # z = (mask_discrete[iter] == 0).sum().item()
        total_ele += t
        ones += o
    ratio_one = ones/total_ele
    return ratio_one

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def compute_mask(model,epoch, temp_increase, args):
#     for m in model.mask_modules:
#         # m.mask_discrete = torch.bernoulli(m.mask)
#         # m.sampled_iter += m.mask_discrete
#         m.temp_s = temp_increase**epoch
#         if epoch == 201:
#             # m.sampled_iter = torch.ones(args.Nbits).cuda()
#             m.temp_s = torch.ones(args.Nbits).cuda() # rewind
        # print('sample_iter:', m.sampled_iter.tolist(), '  |  temp_s:', [round(item,3) for item in m.temp_s.tolist()])
        # if epoch in [args.epochs/2, args.epochs] :
        #     print('sample_iter:', m.sampled_iter.tolist(), '  |  temp_s:', [round(item,3) for item in m.temp_s.tolist()])

def tiny_loader(args):
    # data_dir = '/home/xiaolirui/datasets/tiny-imagenet-200'
    normalize = transform.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transform.Compose(
        [transform.RandomResizedCrop(224), transform.RandomHorizontalFlip(), transform.ToTensor(),
         normalize, ])
    transform_test = transform.Compose([transform.Resize(224), transform.ToTensor(), normalize, ])
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'train'), transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=os.path.join(args.data, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    return train_loader, test_loader

def cifar_loader(args):
    #prepare dataset and preprocessing
    transform_train = transform.Compose([
        transform.RandomCrop(32, padding=4),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)
    return trainloader,testloader

def imagenet_loader(args):
    # Data loading code
    traindir = os.path.join(args.data,'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transform.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transform.Compose([
            transform.RandomResizedCrop(224),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               )

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transform.Compose([
            transform.Resize(256),
            transform.CenterCrop(224),
            transform.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             )
    return train_loader, val_loader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if args.seed is not None:
    random.seed(3407)
    torch.manual_seed(3407)
    cudnn.deterministic = True

    today=datetime.date.today()
    formatted_today=today.strftime('%m%d')
    root =  os.path.join('train_result',formatted_today)
    save_dir =os.path.join(root,args.save_file)
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    writer = SummaryWriter(save_dir)

    train_log_filepath = os.path.join(save_dir, args.log_file)
    logger = get_logger(train_log_filepath)
    logger.info("args = %s", args)

    #prepare dataset and preprocessing
    if args.classes == 10:
        trainloader, testloader = cifar_loader(args)
    elif args.classes == 200:
        trainloader, testloader = tiny_loader(args)
    elif args.classes == 1000:
        trainloader, testloader = imagenet_loader(args)

    #labels in CIFAR10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #define ResNet20
    print("=> creating model '{}'".format(args.arch))
    model = eval(args.arch)(
        num_classes=args.classes,
        Nbits=args.Nbits,
        act_bit = args.act,
        bin=True
        ).to(device)


    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    # optimizer = optim.Adam(model.parameters(),lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.epochs), T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)
    
    logger.info('start training!')
    best_acc = 0
    solid_best_acc =0
    temp_increase = args.final_temp**(1./(args.rewind))
    for epoch in range(1, args.epochs):
        print('\nEpoch: %d' % epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        if args.warmup:
            if epoch <= 5:
                step = epoch/5
                lr = args.lr * step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()
                # adjust_learning_rate(optimizer, epoch, args)
        else:
            if epoch > 1:
                scheduler.step()
                # adjust_learning_rate(optimizer, epoch, args)

        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        
        # update global temp
        if epoch <= args.rewind:
            model.temp = temp_increase**epoch
        else:
            _epoch = epoch - args.rewind # rewind tempreture to 1
            model.temp = temp_increase**_epoch
        logger.info('Current global temp:%.3f'% round(model.temp,3))

        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)

            # get sparsity of network at current epoch

            # Budget-aware adjusting lmbda according to Eq(4)
            ratio_one = get_ratio_one(model)
            TS = args.target / args.Nbits  # target ratio of ones of masks in the network
            regularization_loss = 0
            for m in model.mask_modules:
                regularization_loss += torch.sum(torch.abs(m.mask).sum())
            classify_loss = criterion(outputs, labels)
            loss = classify_loss + (args.lmbda*(ratio_one-TS)) * regularization_loss          
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print ac & loss in each batch
            lrr = optimizer.state_dict()['param_groups'][0]['lr']
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc = 100. * correct / total
            if i % 100 == 0:
                logger.info('Epoch:[{}]\t lr={:.4f}\t Ratio_ones={:.5f}\t loss={:.5f}\t acc={:.3f}'.format(epoch,lrr,ratio_one,sum_loss/(i+1),train_acc ))
            writer.add_scalar('train loss', sum_loss / (i + 1), epoch)
            writer.add_scalar('train loss', ratio_one, epoch)

        # test with soft mask
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_acc = (100 * correct / total)
            logger.info('Test\'s ac is: %.3f%%' % test_acc )
            writer.add_scalar('Test Acc', test_acc, epoch)
        
        if epoch >= args.solidize:
            # Turn soft mask to discrete
            for m in model.mask_modules:
                m.mask= torch.where(m.mask >= 0.5, torch.full_like(m.mask, 1), m.mask)
                m.mask= torch.where(m.mask < 0.5, torch.full_like(m.mask, 0), m.mask)
                # m.mask_discrete = torch.bernoulli(m.mask)
                # logger.info(m.mask)
                logger.info(m.mask)
            # test again after finalizing the soft bitmask to 0&1
            with torch.no_grad():
                _correct = 0
                _total = 0
                for data in testloader:
                    model.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, _predicted = torch.max(outputs.data, 1)
                    _total += labels.size(0)
                    _correct += (_predicted == labels).sum()
                    _test_acc = (100 * _correct / total)
                logger.info('Solid Test\'s ac is: %.3f%%' % _test_acc )
            ratio_one = get_ratio_one(model)
            solid_best_model_path = os.path.join(*[save_dir, 'solid_model_best.pt'])
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_acc': _test_acc,
                'solid_ratio_one': ratio_one,
            }, solid_best_model_path)
            solid_best_acc = _test_acc
            _best_epoch = epoch+1
            avg_bit_ = ratio_one * args.Nbits
            logger.info('Solid Accuracy is %.3f%% , average bit is %.2f%% at epoch %d' %  (solid_best_acc, avg_bit_, _best_epoch))

        # update temp_s based on sampled_iter per epoch
        # if epoch <= 400:
        #     compute_mask(model,epoch, temp_increase, args)
        
    TP = model.total_param()
    avg_bit = args.Nbits * ratio_one
    logger.info('model size is: %.3f' % TP)
    logger.info('average bit is: %.3f ' % avg_bit)
    # plt.plot(np.arange(len(lr)), lr)
    # plt.savefig('learning rate.jpg')
