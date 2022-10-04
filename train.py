import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torchvision import transforms

import tensorboard_logger as tb_logger

from model import VGG
from dataset import TrainDataset, TestDataset
from option import TrainOption
from utils import adjust_learning_rate, accuracy, AverageMeter

def main():
    opt = TrainOption().parse()
    
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    # use the ImageNet transformation
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # define the dataloader to load that single image
    train_loader = DataLoader(dataset=TrainDataset(), shuffle=False, batch_size=1)
    val_loader = DataLoader(dataset=TestDataset(), shuffle=False, batch_size=1)
    
    # initialize the model
    model = VGG()
    
    criterion = nn.CrossEntropy()
    
    # optimizer
    optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        
    # train routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training student...")
        
        # train step
        time1 = time.time()
        
        train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion)
        
        time2 = time.time()
        
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
                
        # validation step
        test_loss, test_acc = validate(epoch, model, val_loader, criterion)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc)

def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    criterion.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index, contrast_idx = data
        
        input = input.cuda()
        target = target.cuda()
        index = index.cuda()
        contrast_idx = contrast_idx.cuda()
        
        out = model(input)
        ce_loss = criterion(out, target)
        
        loss = ce_loss
                
        acc1, acc5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.sum().item(), input.size(0))
        top1.update(acc1[0], input.size(0))
            
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # time meters
        end = time.time()
        batch_time.update(time.time() - end)
        
        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1))
            sys.stdout.flush()
    
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return losses.avg, top1.avg

def validate(epoch, model, val_loader, criterion):
    model.eval()
        
    losses = AverageMeter()
    top1 = AverageMeter()
        
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            input, target = data
        
            input = input.cuda()
            target = target.cuda()
            
            out = model(input)
                
            ce_loss = criterion(out, target)
            
            loss = ce_loss
            
            acc1, acc5 = accuracy(out, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            
            # print info
            if idx % 100 == 0:
                print('Epoch: [{epoch}][{idx}/{3}]\t',
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, idx, len(val_loader), loss=losses, top1=top1))
                sys.stdout.flush()
                
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()