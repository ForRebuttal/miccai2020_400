import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
        
class MultipleLoss(nn.Module):
    def __init__(self, alpha=1):
        super(MultipleLoss, self).__init__()
        self.Loss_main = nn.CrossEntropyLoss()
        self.Loss_att = nn.ModuleList([
                            nn.CrossEntropyLoss()
                            for typeIndex in range(4)
                            ])
                            
        self.alpha = alpha

        
    def forward(self, outputs, targets):
        loss_att = []
        target = targets[:,0:1].view(-1)
        loss_main = self.Loss_main(outputs[0],target)
        
        for i in range(1, len(outputs)):
            target = targets[:,i:i+1].view(-1)
            loss_att.append(self.Loss_att[i-1](outputs[i],target))

        loss = loss_main

        for i in range(len(outputs)-1):
            loss+= self.alpha/len(outputs)*loss_att[i]
            
        return loss
        
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
        
        