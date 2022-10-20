import argparse
import os
import random
import shutil
import time
import datetime
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from imagenet.networks.resnetcs18 import ResNet18
from imagenet.networks.resnetcs50 import ResNet50
from imagenet.networks.vgg import VGG19bn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',
                    metavar='DIR',
                    default='/home/xiaolirui/workspace/datasets/tiny-imagenet-200',
                    help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    help='default: resnet18')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1024,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.05,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=200,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--classes', type=int, default=200, help='class of output')
parser.add_argument('--lmbda', type=float, default=0.001, help='lambda for L1 mask regularization (default: 1e-8)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--act', type=int, default=0, help='quantization bitwidth for activation')
parser.add_argument('--target', type=int, default=3, help='Target Nbit')
parser.add_argument('--Nbits', type=int, default=6, help='quantization bitwidth for weight')

parser.add_argument('--warmup',dest='warmup',action='store_true',help='warmup learning rate for the first 5 epochs')
parser.add_argument('--save_file', type=str, default='TIM_CSQvgg19bn_T6N3A0_lr005', help='path for saving trained models')
parser.add_argument('--log_file', type=str, default='train.log', help='save path of weight and log files')

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
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

    dist.init_process_group(backend='nccl')
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
        # print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
    train_log_filepath = os.path.join(save_dir, args.log_file)
    logger = get_logger(train_log_filepath)
    logger.info("args = %s", args)

    model = eval(args.arch)(
        num_classes=args.classes,
        Nbits=args.Nbits,
        act_bit = args.act,
        bin=True
        )

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      find_unused_parameters=True,
                                                      )

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.epochs/2), T_mult=1, eta_min=0, last_epoch=-1, verbose=False)

    cudnn.benchmark = True

    if args.classes == 1000:
        train_sampler, val_sampler, train_loader, val_loader = imagenet_loader(args)
    elif args.classes == 200:
        train_sampler, val_sampler, train_loader, val_loader = tiny_loader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args, logger)
        return
    
    temp_increase = 200**(1./(args.epochs*0.6))
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, args)
        if args.warmup:
            if epoch <= 5:
                step = epoch/5
                lr = args.lr * step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()
        else:
            if epoch > args.start_epoch:
                scheduler.step()
        
        # update global temp
        if epoch <= args.epochs/2:
            model.temp = temp_increase**epoch
        else:
            _epoch = epoch - (args.epochs/2)
            model.temp = temp_increase**_epoch

        # train for one epoch
        ratio_one = get_ratio_one(model)
        logger.info('Current R_O:%.3f'% round(ratio_one,3))
        train(train_loader, model, criterion, optimizer, epoch, local_rank, args, logger)  
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args, logger)

        # validate again based on solid bit mask
        if epoch > args.epochs*0.975:
            for m in model.module.mask_modules:
                m.mask= torch.where(m.mask >= 0.5, torch.full_like(m.mask, 1), m.mask)
                m.mask= torch.where(m.mask < 0.5, torch.full_like(m.mask, 0), m.mask)
                logger.info(m.mask_discrete)
            solid_acc1 = validate(val_loader, model, criterion, local_rank, args, logger)
            logger.info('Solid Test\'s ac is: %.3f%%' % solid_acc1 )
            ratio_one = get_ratio_one(model)
            solid_best_model_path = os.path.join(*[save_dir, 'solid_model_best.pt'])
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_acc': solid_acc1,
                'solid_ratio_one': ratio_one,
            }, solid_best_model_path)
            solid_best_acc = solid_acc1
            _best_epoch = epoch+1
            avg_bit_ = ratio_one * args.Nbits
            logger.info('Solid Accuracy is %.3f%% , average bit is %.2f%% at epoch %d' %  (solid_best_acc, avg_bit_, _best_epoch))

        if epoch <= args.epochs*0.95:
            compute_mask(model, epoch, temp_increase, args)
            
    avg_bit = args.Nbits * ratio_one
    logger.info('average bit is: %.3f ' % avg_bit)



def train(train_loader, model, criterion, optimizer, epoch, local_rank, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # update global tempreture
    temp_increase = 200**(1./(args.epochs/2))
    if epoch <= args.epochs/2:
            model.temp = temp_increase**epoch
    
    logger.info('Current global temp:%.3f'% round(model.temp,3))

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        
        ratio_one = get_ratio_one(model)
        # logger.info('Current R_O:%.3f'% round(ratio_one,3))

        # Budget-aware adjusting lmbda according to Eq(4)
        TS = args.target / args.Nbits  # target ratio of ones of masks in the network
        regularization_loss = 0
        for m in model.module.mask_modules:
            regularization_loss += torch.sum(torch.abs(m.mask).sum())
        
        classify_loss = criterion(output, target)
        loss = classify_loss + (args.lmbda*(ratio_one-TS)) * regularization_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, local_rank, args,logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
        #                                                             top5=top5))
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_mask(model,epoch, temp_increase, args):
    for m in model.mask_modules:
        m.mask_discrete = torch.bernoulli(m.mask)
        m.sampled_iter += m.mask_discrete
        m.temp_s = temp_increase**m.sampled_iter
#         if epoch == args.epochs/2:
#             m.sampled_iter = torch.ones(args.Nbits)
#             m.temp_s = torch.ones(args.Nbits)
        print('sample_iter:', m.sampled_iter.tolist(), '  |  temp_s:', [round(item,3) for item in m.temp_s.tolist()])

def get_ratio_one(model):
    mask_discrete = [m.mask_discrete for m in model.module.mask_modules]
    total_ele = 0
    ones = 0
    for iter in range(len(mask_discrete)):
        t = mask_discrete[iter].numel()
        o = (mask_discrete[iter] == 1).sum().item()
        # z = (mask_discrete[iter] == 0).sum().item()
        total_ele += t
        ones += o
    ratio_one = ones/total_ele
    return ratio_one


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

def tiny_loader(args):
    # data_dir = '/home/xiaolirui/datasets/tiny-imagenet-200'
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(0.5), 
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(args.data, 'train'), transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    
    valset = datasets.ImageFolder(root=os.path.join(args.data, 'val'), transform=transform_test)
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    return train_sampler, val_sampler, train_loader, val_loader

def imagenet_loader(args):
    # Data loading code
    traindir = os.path.join(args.data,'train')
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=4,
                                             pin_memory=True,
                                             sampler=val_sampler)
    return train_sampler, val_sampler, train_loader, val_loader

if __name__ == '__main__':
    main()
