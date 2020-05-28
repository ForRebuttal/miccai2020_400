import argparse
import logging
import os
import sys
import gc
import cv2

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from utils import ThyroidDataset

from torch import optim
from torch.autograd import Variable
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


class GradCam:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def __call__(self, input, classifier_index, index=None):
        # if self.cuda:
        #     features, output = self.extractor(input.cuda(), vector.cuda(), mask_nodule.cuda(), mask_margin.cuda(), aspect.cuda(), classifier_index)#self.forward(input.cuda())
        # else:
        #     features, output = self.extractor(input, vector, mask_nodule, mask_margin, aspect, classifier_index)#self.forward(input)
        #

        input = torch.from_numpy(input.transpose(2, 0, 1)).unsqueeze(0)
        if self.cuda:
            input = input.float().cuda()

        mask_pred, outputs, features = self.model(input)
        output = outputs[classifier_index+1]

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        pre = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.model.gradients.cpu().data.numpy()

        target = features
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam, pre, output, mask_pred

def show_cam_on_image(img, mask, size, rootPath):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (size[0], size[1]))
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    result = np.uint8(255 * cam)
    print(rootPath+"cam.jpg")
    cv2.imwrite(rootPath+"cam.jpg", result)

def test_single(net, device, fname):
    img_pth = "./demo/demo_img/{}".format(fname)
    vis_pth = "./demo/visualization/{}".format(fname.split(".")[0])
    try:
        os.mkdir(vis_pth)
    except OSError:
        pass
    img = Image.open(img_pth)
    orig_size = img.size
    np_img = np.array(img.resize((224,224)))/255
    #print(np_img.shape)
    mask_np = eval_demo(net, np_img, device, log.logger)
    print("Start generate segmentation...")
    mask_img = Image.fromarray(mask_np.astype('uint8')).convert('RGB').resize(orig_size)
    mask_img.save(os.path.join(vis_pth, fname.split(".")[0]+"_seg.jpg"))
    print("Finished.")
    print("Start generate heatmap for attributes...")
    attribute_name = ["Echogenic Foci", "Composition", "Margin", "Echogenicity"]
    target_name = [["none", "exists"], ["solid", "non-solid"],
                   ["ill-defined", "well-defind"], ["hypoechogenic", "hyper-isoechogenic"]]
    typeList = [2, 2, 2, 2]
    grad_cam = GradCam(model=net, use_cuda=True)
    for class_index in range(0, len(typeList)):
        for target_index in range(0, typeList[class_index]):
            cam_mask, output_index, output, mask_pred = grad_cam(np_img, class_index, target_index)
            rootPath = os.path.join(vis_pth, fname.split(".")[0])
            rootPath += "_" + attribute_name[class_index] + '_' + target_name[class_index][target_index] + '_'
            show_cam_on_image(img, cam_mask, orig_size, rootPath)
    print("finished.")

    #log.logger.info('Validation Dice Coeff: {}'.format(val_score))



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename', dest='fname', type=str, default="demo1.jpg",
                        help='The file in ./demo/demo_img/')

    return parser.parse_args() 


if __name__ == '__main__':

    log = Logger("log_demo.txt")

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
    
    print("Filename: {}".format(args.fname))
    try:
        test_single(net, device, args.fname)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
