import torch
import torch.nn as nn
import torch.nn.functional as F
        
class MultipleLoss(nn.Module):
    def __init__(self):
        super(MultipleLoss, self).__init__()
        self.Loss_main = nn.CrossEntropyLoss()
        self.Loss_att = nn.ModuleList([
                            nn.CrossEntropyLoss()
                            for typeIndex in range(4)
                            ])

        
    def forward(self, outputs, targets):
        loss_att = []
        target = targets[:,0:1].view(-1)
        loss_main = self.Loss_main(outputs[0],target)
        
        for i in range(1, len(outputs)):
            target = targets[:,i:i+1].view(-1)
            loss_att.append(self.Loss_att[i-1](outputs[i],target))

        loss = loss_main

        for i in range(len(outputs)-1):
            loss+=(1.0/len(outputs))*loss_att[i]
            
        return loss
        
        