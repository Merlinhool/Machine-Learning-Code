import argparse
import os
import shutil
import time
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#import ResNet
import ResNet as ResNet


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

args = parser.parse_args()

args.depth = 164
args.lr = 0.1 if args.depth < 100 else 0.01
args.num_iteration = 80000 if args.depth < 100 else 90000
args.weight_decay_point = {400:10, 32000:0.1, 48000:0.1, 64000:0.5, 80000:0.5} if args.depth >= 100 else {32000:0.1, 48000:0.1, 64000:0.2}

args.momentum = 0.9
args.weight_decay = 0.0001
args.batch_size = 128
args.is_nesterov = True
args.data = '../cifar_10_data/'
args.class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
args.pre_evaluate = False
args.val_train_rate = 0.1
args.train_mean_path = '../cifar10-per-pixel-mean.pt'
args.best_model_path = 'best_model.pt'
args.tmp_model_path = 'tmp_model.pt'
args.resume = False
args.resume_model_path = args.best_model_path
args.device_id = 0
time_file = open('time_log', 'w')
loss_output = dict()
train_prec1_output = dict()
val_prec1_output = dict()
test_prec1_output = dict()


#TODO
#args.print_freq = args.batch_size
args.print_freq = 1000
args.pin_memory = True
#use_cuda = False
use_cuda = torch.cuda.is_available()

# TODO
print 'Loading train mean ...'
train_mean = torch.load(args.train_mean_path)
train_mean = train_mean.repeat(3,1,1)
print 'train mean:'
print train_mean
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x, mean=train_mean: x - mean)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x, mean=train_mean: x - mean)
])

'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''

#print args

def main():
    global args

    best_prec1 = 0
    start_iteration = 0

    print('==> Building model ...')
    model = ResNet.resnet_bottleneck(args.depth, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.is_nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume_model_path):
            print("=> loading checkpoint '{}'".format(args.resume_model_path))
            checkpoint = torch.load(args.best_model_path)
            args.start_iteration = checkpoint['iteration']
            best_prec1 = checkpoint['best_prec1']
            args.lr = checkpoint['learning_rate']
            model.load_state_dict(OrderedDict([(k[7:],v) for (k,v) in checkpoint['model_state_dict']]))
            optimizer.load_state_dict(OrderedDict([(k[7:],v) for (k,v) in checkpoint['optimizer']]))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_model_path))

    if use_cuda:
        print('using cuda ...')
        criterion = criterion.cuda()
        torch.cuda.set_device(args.device_id)
        model.cuda()
        #tn = torch.cuda.device_count()
        #model = torch.nn.DataParallel(model, device_ids=args.device_id)
        print('using ' + str(args.device_id) + ' GPUs ...')
        cudnn.benchmark = True

    # Data loading code
    print('==> Preparing data..')
    '''
    train_val_ds = datasets.CIFAR10(root=args.data, train=True, download=False, transform=transform_train)
    train_ds, val_ds = validation_split(train_val_ds, args.val_train_rate)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = args.pin_memory)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = args.pin_memory)

    test_ds = datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory = args.pin_memory)
    '''
    train_val_ds = datasets.CIFAR10(root=args.data, train=True, download=False, transform=transform_train)
    train_ds, _ = validation_split(train_val_ds, args.val_train_rate)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = args.pin_memory)

    train_val_ds = datasets.CIFAR10(root=args.data, train=True, download=False, transform=transform_test)
    _, val_ds = validation_split(train_val_ds, args.val_train_rate)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = args.pin_memory)

    test_ds = datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory = args.pin_memory)
    '''
    train_ds = datasets.CIFAR10(root=args.data, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = args.pin_memory)
    val_ds = datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory = args.pin_memory)
    '''

    print 'train_loader len:', len(train_loader)
    print 'val_loader len:', len(val_loader)
    print 'test_loader len:', len(test_loader)
    
    classes = args.class_names

    if args.pre_evaluate:
        validate(val_loader, model, criterion, best)
        return
    
    train(train_loader, val_loader, test_loader, model, criterion, optimizer, best_prec1, start_iteration)

    val_prec1 = validate(val_loader, model, criterion)
    test_prec1 = validate(test_loader, model, criterion)
    print 'last model:\nval prec@1 =', val_prec1, '\ttest prec@1 =', test_prec1

    checkpoint = torch.load(args.best_model_path)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['model_state_dict'])

    val_prec1 = validate(val_loader, model, criterion)
    test_prec1 = validate(test_loader, model, criterion)
    print 'best model:\nrecorded val prec@1 =', best_prec1, '\trecalculated val prec@1 =', val_prec1, '\ttest prec@1 =', test_prec1

def train(train_loader, val_loader, test_loader, model, criterion, optimizer, best_prec1=0, iteration=0):
    global args

    # switch to train mode
    model.train()
    start_time = time.time()
    epoch = 0
    learning_rate = args.lr

    while iteration < args.num_iteration:
        epoch += 1
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            iteration += 1
            learning_rate = adjust_learning_rate(optimizer, iteration, learning_rate)

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                input = input.cuda(async=True)
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            '''
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            '''

            prec = accuracy(output.data, target)
            prec1 = prec[0]
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        val_prec1 = validate(val_loader, model, criterion)
        test_prec1 = validate(test_loader, model, criterion)

        is_best = False
        if val_prec1 > best_prec1:
            is_best = True
            best_prec1 = val_prec1

        save_checkpoint({
            'iteration': iteration,
            'epoch': epoch,
            #'arch' : args.arch,
            'model_state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'learning_rate': learning_rate,
        }, is_best)

        print('Epoch: [{0}] [{1}/{2}] (total iter: [{3}])\n'
              'Time batch:{batch_time.avg:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Prec@1 train:{top1.avg:.3f}, val: {val_prec1:.3f}, test: {test_prec1:.3f}\t'.format(
               epoch, 0, len(train_loader), iteration, batch_time=batch_time, loss=losses, top1=top1, val_prec1=val_prec1, test_prec1=test_prec1))
        time_file.write('Epoch: [{0}] [{1}/{2}] (total iter: [{3}])\n'
              'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\n'
              'Data {data_time.val:.3f} (avg: {data_time.avg:.3f})\n\n'.format(
               epoch, i, len(train_loader), iteration, batch_time=batch_time,
               data_time=data_time))
        
        loss_output[iteration] = losses.avg
        train_prec1_output[iteration] = top1.avg
        val_prec1_output[iteration] = val_prec1
        test_prec1_output[iteration] = test_prec1

    tot_time = int(round(time.time() - start_time))
    if tot_time < 60:
        time_output = str(tot_time) + ' s'
    elif tot_time < 3600:
        time_output = str(tot_time // 60) + ' m ' + str(tot_time%60) + ' s'
    else:
        time_output = str(tot_time // 3600) + ' h ' + str((tot_time % 3600) // 60) + ' m ' + str(tot_time % 60) + ' s'

    toJson(loss_output, 'loss.json')
    toJson(train_prec1_output, 'train_prec1.json')
    toJson(val_prec1_output, 'val_prec1.json')
    toJson(test_prec1_output, 'test_prec1.json')

    print 'total iteration:', iteration
    print 'total epoch:', epoch
    print 'total train time:', time_output
    print 'best val prec@1:', best_prec1


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        if use_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        '''
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        '''
        prec = accuracy(output.data, target)
        prec1 = prec[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    model.train()
    return top1.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[i+self.offset]

def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).
    
       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds
       
       """
    val_offset = int(len(dataset)*(1-val_share))
    print 'total data num =', len(dataset)
    print 'train num =', val_offset
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset)-val_offset)


def adjust_learning_rate(optimizer, iteration, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    flag = False
    if iteration in args.weight_decay_point:
        learning_rate *= args.weight_decay_point[iteration]
        flag = True
        print 'Learning Rate change!\t iteration:', iteration, '\t learning_rate:', learning_rate

    for param_group in optimizer.param_groups:
        if flag:
            print param_group['lr']
        param_group['lr'] = learning_rate
    return learning_rate


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename=args.tmp_model_path):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.best_model_path)

def toJson(output, path):
    f = open(path, 'w')
    json.dump(output, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()
    time_file.close()
