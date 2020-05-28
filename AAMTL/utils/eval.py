import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from utils.loss import dice_coeff
import numpy as np
def calc_single_result(output, target, count_tot):
    curr_max = torch.max(output.cpu().data).item()
    index = target.cpu().data[0].item()              #actually right
    count_tot[index]+=1
    # outnp=output[:,0:2].cpu().data.numpy()[0]
    # for j in range(2):
    #     if outnp[j] == curr_max:
    #         pred = j

    _, pred = torch.max(output, 1)

    return index, pred.item(), count_tot, curr_max

def Weighted_Error(matrix):
    acc=0
    n = len(matrix)
    for i in range(n):
        acc+=matrix[i][i]/sum(matrix[i])/n
    return 1-acc

def F1_score(matrix):
    precision = matrix[0][0]/(matrix[0][0]+matrix[0][1])
    recall = matrix[0][0]/(matrix[0][0]+matrix[1][0]+0.0001)
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
    
def result_stat(output, target, matrix, count_tot, count_right, tot, incorrect):
    index, pred, count_tot, curr_max = calc_single_result(output, target, count_tot)
    matrix[index][pred] += 1
    if pred == index:
        count_right[index] += 1
        tot += 1
    else:
        incorrect += 1
        
    return matrix, count_tot, count_right, tot, incorrect
        
def result_output(logger, loader, tot, matrix, count_tot, count_right, incorrect, name):
    nTotal = len(loader.dataset)
    err = 100. * incorrect / nTotal
    w_err = Weighted_Error(matrix)
    precision, recall, F1 = F1_score(matrix)
    logger.info("==={}===".format(name))
    logger.info('Error: {}/{} ({:.2f}%)'.format(incorrect, nTotal, err))
    logger.info('Weighted Error: ({:.2f}%)'.format(100 * w_err))
    logger.info(count_right)
    logger.info(count_tot)
    logger.info(matrix)
    logger.info("{}/{}".format(tot, nTotal))
    logger.info("Precision:{:.4f}\tRecall:{:.4f}\tF1_Score:{:.4f}".format(precision, recall, F1))
    logger.info("kappa:{:.4f}\n".format(kappa_value(matrix)))

def eval_net(net, loader, device, n_val, logger):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    
    Att_Name = ["Calc", "Comp", "Marg", "Echo"]
    
    dice_tot = 0
    tot = 0
    incorrect = 0
    count_right = [0]*2
    count_tot = [0]*2
    matrix = [[0] * 2 for row in range(2)]

    incorrect_att =   []
    count_right_att = []
    count_tot_att =   []
    matrix_att =      []
    tot_att = []
    
    for i in range(4):
        tot_att.append(0)
        incorrect_att.append(0)
        count_right_att.append([0]*2)
        count_tot_att.append([0]*2)
        matrix_att.append([[0] * 2 for row in range(2)])
        
        

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            target = batch['target']
            aspect = batch['aspect']
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            target = target.to(device=device, dtype=torch.long)
            aspect = aspect.to(device=device, dtype=torch.float32)

            mask_pred, outputs, _ = net(imgs)
            #mask_pred = net(imgs, aspect)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                if net.n_classes > 1:
                    dice_tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
                else:
                    dice_tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])
            
            #main task
            matrix, count_tot, count_right, tot, incorrect = result_stat(outputs[0], target[:, 0:1].view(-1), matrix, count_tot, count_right, tot, incorrect)
            #attribute tasks
            for i in range(4):
                matrix_att[i], count_tot_att[i], count_right_att[i], tot_att[i], incorrect_att[i] = result_stat(outputs[i+1], target[:, i+1:i+2].view(-1), 
                                                                      matrix_att[i], count_tot_att[i], count_right_att[i], tot_att[i], incorrect_att[i])
                
        result_output(logger, loader, tot, matrix, count_tot, count_right, incorrect, "Tot")
        for i in range(4):
            result_output(logger, loader, tot_att[i], matrix_att[i], count_tot_att[i], count_right_att[i], incorrect_att[i], Att_Name[i])
              
    return dice_tot / n_val, kappa_value(matrix)
