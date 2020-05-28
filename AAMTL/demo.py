import argparse
import logging
import os
import sys
import gc

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from utils import ThyroidDataset

from torch import optim
from tqdm import tqdm

from utils.ThyroidDataset import BasicDataset
from utils.eval_demo import eval_demo
from utils.loss import MultipleLoss

from network import densenet

import torchvision.models as models
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

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

def test_single(net, device, img_pth):
    
    img = Image.open(img_pth)
    np_img = np.array(img.resize((224,224)))/255   
    #print(np_img.shape)
    eval_demo(net, np_img, device, log.logger)
        
    #log.logger.info('Validation Dice Coeff: {}'.format(val_score))



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename', dest='fname', type=str, default="demo1.jpg",
                        help='The file in ./demo/demo_img/')

    return parser.parse_args() 


if __name__ == '__main__':

    log = Logger("demo.txt")

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1
    torch.manual_seed(seed)
    if device=="cuda":
        torch.cuda.manual_seed(seed)    

    log.logger.info(f'Using device {device}')

    log.logger.info(f'Start initialize model...')
    
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
    log.logger.info(f'Model loaded from {model_dir}')
    net.load_state_dict(
        torch.load(model_dir, map_location=device)
    )
    
    log.logger.info(f'Model initialized')
    net.to(device=device)    
    img_pth = "./demo/demo_img/{}".format(args.fname)
    print("Filename: {}".format(args.fname))
    try:
        test_single(net, device, img_pth)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
