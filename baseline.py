# -*- coding: utf-8 -*-
import argparse
import sys
import os
import time
from datetime import datetime, timedelta, timezone
import datetime
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import PolynomialLR
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import functional as F
from utils.utils import get_model, get_transforms, get_dataset, print_header
from optimizers import *

from torch.optim.swa_utils import AveragedModel

from lightning.fabric import Fabric

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--log', action='store_false', default=True, help='save log')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_worker', default=8, type=int, help='number of workers')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-3, type=float)
parser.add_argument('--optimizer',default='SGD',type=str,help='SGD')
parser.add_argument('--model',default='ResNet18',type=str)
parser.add_argument('--milestones', default=[60, 120, 160], nargs='*', type=int, help='milestones of scheduler')
parser.add_argument('--dataset',default='CIFAR10',type=str)
parser.add_argument('--lr_decay',default=0.2,type=float)
parser.add_argument('--lr_type',default='MultiStepLR',type=str)
parser.add_argument('--eta_min',default=0.00,type=float)
# for Averaging
parser.add_argument('--start_averaged', default=160, type=int)

args = parser.parse_args()

# 出力先を標準出力orファイルで選択
if args.log is True:
    today = datetime.now(timezone(timedelta(hours=+9))).strftime("%Y-%m-%d")
    log_dir = f"./light_logs/{args.dataset}/{args.model}/{today}/SGD"
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=+9))).strftime("%H%M")
    logpath = log_dir+f"/{now}-{args.epoch}-{args.lr_type}-{args.weight_decay:.0e}.log"
    sys.stdout = open(logpath, "w") # 表示内容の出力をファイルに変更

print(' '.join(sys.argv))
print('GPU count',torch.cuda.device_count())
print('epoch:', args.epoch)
print('model:',args.model)
print('dataset:',args.dataset)
print('batch_size:', args.batch_size)
print('optimizer:',args.optimizer)
print('learning_rate:', args.lr)
print('lr_decay:',args.lr_decay)
print('lr_type:',args.lr_type)
print('milestones:', args.milestones)
print('weight_decay:',args.weight_decay)
print('momentum:', args.momentum)
print('eta_min:',args.eta_min)
print('start_averaged:',args.start_averaged)

fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=torch.cuda.device_count())
fabric.launch()

#define dataset and transform
transform_train, transform_test = get_transforms(args.model, args.dataset)
trainset, testset = get_dataset(args.dataset, transform_train, transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_worker)

trainloader, testloader = fabric.setup_dataloaders(trainloader, testloader)

num_class = len(trainset.classes)

#define model
print('==> Building model..')
model = get_model(args.model, num_class, args.dataset)
averaged_model = AveragedModel(model)
torch.cuda.empty_cache()
cudnn.benchmark = True
print('==> Finish model')

#lr decay milestones
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.lr_type == 'MultiStepLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, milestones=args.milestones, gamma=args.lr_decay)
elif args.lr_type == 'CosineAnnealingLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, T_max = args.epoch, eta_min = args.eta_min)

model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler)
criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    torch.cuda.synchronize()
    time_ep = time.time()
    model.train()
    averaged_model.train()
    train_loss = 0.0
    correct = 0   
    total = 0

    for batch_idx, (input, target) in enumerate(trainloader):
        input_size = input.size()[0]
        
        output = model(input)
        total += input_size
        loss = criterion(output, target)
        optimizer.zero_grad()
        fabric.backward(loss)
        optimizer.step()

        # update Averaged model
        if epoch >= args.start_averaged:
            averaged_model.update_parameters(model)

        with torch.no_grad():
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum()

    if epoch >= args.start_averaged:
        torch.optim.swa_utils.update_bn(trainloader, averaged_model, optimizer.param_groups[0]["params"][0].device)
    
    torch.cuda.synchronize()   
    time_ep = time.time() - time_ep

    return train_loss/batch_idx, 100*correct/total, time_ep

def test(epoch, model):
    model.eval()
    total = 0

    test_loss = 0.0
    test_correct = 0
    ave_test_loss = 0.0
    ave_test_correct = 0.0
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testloader):
            input_size = input.size()[0]  
            # normal test
            output = model(input)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = torch.max(output, 1)
            test_correct += pred.eq(target).sum()
            # averaged test
            if epoch >= args.start_averaged:
                ave_output = averaged_model(input)
                loss = criterion(ave_output, target)
                ave_test_loss += loss.item()
                _, pred = torch.max(ave_output, 1)
                ave_test_correct += pred.eq(target).sum()
            total += input_size
    return test_loss/batch_idx, 100*test_correct/total, ave_test_loss/batch_idx, 100*ave_test_correct/total

total_time = 0
for epoch in range(args.start_epoch, args.epoch):
    if epoch == 0:
        print_header()
    train_loss, train_acc, time_ep = train(epoch)
    total_time += time_ep
    if not args.lr_type == "fixed":
        scheduler.step()
    lr_ = optimizer.param_groups[0]['lr']
    test_loss, test_acc, ave_loss, ave_acc = test(epoch, model)
    fabric.print(f"┃{epoch:12d}  ┃{lr_:12.4f}  │{time_ep:12.3f}  ┃{train_loss:12.4f}  │{train_acc:10.2f} %  "\
          f"┃{test_loss:12.4f}  │{test_acc:10.2f} %  ┃{ave_loss:12.4f}  │{ave_acc:10.2f} %  ┃")

fabric.print('Total {:.0f}:{:.0f}:{:.0f}'.format(total_time//3600, total_time%3600//60, total_time%3600%60))
fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
# ファイルを閉じて標準出力を元に戻す
if args.log is True:
    sys.stdout.close()
    sys.stdout = sys.__stdout__