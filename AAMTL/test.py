import argparse
import logging
import os
import sys
import gc

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
data_dir = "/home/kongming/dagengren/data/Thyroid"
dir_checkpoint = 'checkpoints_src/'
model_dir = "../pretrained_model/model.pth"

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
    # outnp=output[:,0:2].cpu().data.numpy()[0]
    # for j in range(2):
    #     if outnp[j] == curr_max:
    #         pred = j

    _, pred = torch.max(output, 1)

    return index, pred.item(), count_tot, curr_max

def test(net,
         device,
         batch_size,
         lr=0.01,
         save_cp=True):
    
    dataset = BasicDataset(data_dir)
    train, val = dataset.split_train_val(10, 0)
    n_train, n_val = len(train),len(val)
    
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    log.logger.info(f'''Starting training:
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')
        
    val_score, kappa = eval_net(net, val_loader, device, n_val, log.logger)
        
    log.logger.info('Validation Dice Coeff: {}'.format(val_score))



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8, 
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=True,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args() 


if __name__ == '__main__':

    log = Logger("log_test.txt")

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed(seed)    
    
    log.logger.info(f'Using device {device}')

    
    net_pretrained=models.densenet201(pretrained=True)
    
    
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
            #k_S[0]="block_{}".format(block_num)
            k_net = "block_{}.".format(block_num)
            for i in k_S:
                k_net=k_net+i+"."
            k_net = k_net[:-1]

        #print(k)
        #print(k_net)        
            
        v = net_pretrained.features.state_dict()[k]
        net_dict[k_net]=v
        
    net.load_state_dict(net_dict)  
    
    if args.load:
        net.load_state_dict(
            torch.load(model_dir, map_location=device)
        )
        log.logger.info(f'Model loaded from {model_dir}')

    net.to(device=device)    
    
    try:
        test(net=net,
             batch_size=args.batchsize,
             lr=args.lr,
             device=device)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
