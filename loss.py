import torch
import torch.nn as nn
import torch.nn.functional as F



def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size(0)
    true = true.view(batch_size,1)
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    class_labels = torch.arange(no_of_classes).float().cuda()
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    y = nn.Softmax(dim=1)(-phi)
    return y

def loss_function(output, labels, loss_type, expt_type, scale):
    targets = true_metric_loss(labels, expt_type, scale)
    return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()