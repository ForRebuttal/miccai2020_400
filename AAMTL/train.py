import argparse
import logging
import os
import sys
import gc
import configparser

import numpy as np
import torch
import torch.nn as nn

from utils import ThyroidDataset

from torch import optim
from tqdm import tqdm

from utils.ThyroidDataset import BasicDataset
from utils.eval import eval_net
from utils.loss import MultipleLoss

from network import densenet

import torchvision.models as models
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader, random_split

from torch.backends import cudnn

from mylogger import Logger

cudnn.benchmark = False
cudnn.deterministic = True 

config = configparser.ConfigParser()
config.read("config.txt")

data_dir = config.get('path','data_dir')
checkpoint_dir = config.get('path','checkpoint_dir')
logger_file = config.get('path','logger_file')

class settings():
    def __init__(self):
        batchsize = 0
        epochs = 0
        lr = 0
        loss_alpha = 0
        loss_beta = 0

def F1_score(matrix):
    precision = matrix[0][0]/(matrix[0][0]+matrix[0][1])
    recall = matrix[0][0]/(matrix[0][0]+matrix[1][0])
    F1 = 2*precision*recall/max((precision+recall),0.0001)
    return precision, recall, F1

def kappa_value(matrix):
    matrix = np.array(matrix)
    length = matrix.shape[0]
    po = np.sum(matrix.diagonal())/np.sum(matrix)
    pe=0
    for i in range(length):
        pe+=np.sum(matrix[i])*np.sum(matrix[:,i])
    pe/=(np.sum(matrix)**2)
    return (po-pe)/(1-pe)


def Weighted_Error(matrix):
    acc = 0
    n = len(matrix)
    for i in range(n):
        acc += matrix[i][i] / sum(matrix[i]) / n
    return 1 - acc


def calc_single_result(output, target, count_tot):
    curr_max = torch.max(output.cpu().data).item()
    index = target.cpu().data[0].item()  # actually right
    count_tot[index] += 1

    _, pred = torch.max(output, 1)

    return index, pred.item(), count_tot, curr_max
    
def calc_err(output, target, length):
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    incorrect = pred.ne(target).cpu().sum()
    err = 100. * float(incorrect) / length
    
    return err
    
def train_net(args,
              net,
              device,
              n_fold,
              epochs=100,
              batch_size=1,
              lr=0.01,
              save_cp=True):
    
    dataset = BasicDataset(data_dir)
    train, val = dataset.split_train_val(10, n_fold)
    n_train, n_val = len(train),len(val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    log.logger.info(f'''Starting training:
        Fold:            {n_fold}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    if net.n_classes > 1:
        criterion1 = nn.CrossEntropyLoss()
    else:
        criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = MultipleLoss(args.loss_alpha)

    best_kappa = 0
    
    for epoch in range(epochs):
        if epoch==50 or epoch==100:
            lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)
        net.train()

        nProcessed = 0
        tot_loss = 0
        tot_err = 0
        tot_att_err=[0]*4
  
        epoch_loss = 0
        count = 0
        
        Att_Name = ["Calc","Comp","Marg","Echo"]
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                count+=1
                imgs = batch['image']
                true_masks = batch['mask']
                target = batch['target']
                aspect = batch['aspect']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                target = target.to(device=device, dtype=torch.long)
                aspect = aspect.to(device=device, dtype=torch.float32)

                masks_pred, outputs = net(imgs)

                loss1 = criterion1(masks_pred, true_masks)
                loss2 = criterion2(outputs, target)
                loss  = args.loss_beta*loss1+loss2
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss1 (batch)': loss1.item(),'loss2 (batch)': loss2.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                nProcessed += len(imgs)
                
                length = len(imgs)
                
                err = calc_err(outputs[0], target[:, 0:1].view(-1).data, length)
                tot_err += err
                
                for i in range(4):
                    att_err = calc_err(outputs[i+1], target[:, i+1:i+2].view(-1).data, length)
                    tot_att_err[i] += att_err

                tot_loss += loss2.data.item()
                
                pbar.update(imgs.shape[0])
                global_step += 1

            
            tot_loss = tot_loss / (count + 1)
            tot_err /= (count + 1)
            for i in range(4):
                tot_att_err[i] /= (count+1)

            log.logger.info('Train Epoch: {:.2f} \t Loss: {:.6f}\tError: {:.6f}'.format(epoch, tot_loss, tot_err))
            for i in range(4):
                log.logger.info('{} Error: {:.6f}'.format(Att_Name[i], tot_att_err[i]))
            
            val_score, kappa = eval_net(net, val_loader, device, n_val, log.logger)
            
            if kappa>best_kappa:
                log.logger.info("best kappa updated.")
                best_kappa = kappa
                if save_cp:
                    try:
                        os.mkdir(checkpoint_dir)
                        log.logger.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net.state_dict(),
                               checkpoint_dir + f'CV_{n_fold}.pth')
                    log.logger.info(f'Checkpoint {epoch + 1} saved !')

            log.logger.info('Validation Dice Coeff: {}'.format(val_score))
            writer.add_scalar('Dice/test', val_score, global_step)

            writer.add_images('images', imgs, global_step)
            if net.n_classes == 1:
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8, 
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args() 

def from_config(config):
    args = settings()
    args.batchsize  = int(config.get('setting','batchsize'))
    args.epochs     = int(config.get('setting','epoch')) 
    args.lr         = float(config.get('setting','lr'))
    args.loss_alpha = float(config.get('hyperparam','loss_alpha'))
    args.loss_beta  = float(config.get('hyperparam','loss_beta'))
    return args

if __name__ == '__main__':

    log = Logger(logger_file)
    #args = get_args()
    args = from_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed(seed)    
    
    log.logger.info(f'Using device {device}')
    
    net_pretrained=models.densenet201(pretrained=True)
    for i_fold in range(0,10):
        net = densenet.densenet201(num_classes=1)

        net_dict = net.state_dict()

        for k in net_pretrained.features.state_dict():
            k_S = k.split(".")
            block_num=int(k_S[0][-1])
            if block_num==5:
                k_net = "block_4."
                for i in k_S:
                    k_net=k_net+i+"."
                k_net = k_net[:-1]
            else:
                k_net = "block_{}.".format(block_num)
                for i in k_S:
                    k_net=k_net+i+"."
                k_net = k_net[:-1]
       
                
            v = net_pretrained.features.state_dict()[k]
            net_dict[k_net]=v
            
        net.load_state_dict(net_dict)  

        net.to(device=device)    
    
        try:
            train_net(args,
                      net=net,
                      n_fold=i_fold,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            log.logger.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
