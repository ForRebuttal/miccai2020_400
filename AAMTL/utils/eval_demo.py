import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from utils.loss import dice_coeff
import numpy as np

Main=["Malignant", "Benign"]

Attribute = [("Echogenic Foci:",["none", "exists"]), ("Composition:\t",["solid", "non-solid"]), ("Margin:\t\t",["ill-defined", "well-defind"]), ("Echogenicity:\t",["hypoechogenic", "hyper/isoechogenic"])]

def output_result(output):
    #main
    result=output[0].cpu().data
    result=F.softmax(result, dim=1)
    prop, idx = torch.max(result, 1)
    print("Diagnisis:\t\t{}".format(Main[idx]))
    
    for i in range(4):
        result=output[i+1].cpu().data
        result=F.softmax(result, dim=1)
        prop, idx = torch.max(result, 1)
        print("{} \t{}".format(Attribute[i][0], Attribute[i][1][idx]))
    
    


def eval_demo(net, orig_img, device, logger):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    img = torch.from_numpy(orig_img.transpose(2,0,1)).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    #print(img.size())
    Att_Name = ["Calc", "Comp", "Marg", "Echo"]
    mask_pred, outputs = net(img)
    output_result(outputs)
    mask_img_np = (mask_pred.cpu().data.squeeze().numpy()>0.5)*255
    return mask_img_np
    
