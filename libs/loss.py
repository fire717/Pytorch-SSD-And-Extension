
#!-*- coding: utf-8 -*-

from torch import nn
import torch
from torch.nn import functional as F

from libs.box_utils import matchPriorBox

class myLoss(nn.Module):
    def __init__(self):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 阿尔法α,类别权重.      
                    当α是列表时,为各类别权重；
                    当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                    常用于目标检测算法中抑制背景类, 
                    retainnet中设置为0.25
        :param gamma: 伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(myLoss,self).__init__()


    def forward(self, confidence, location, prior_boxes, labels):
        """
        confidence: 预测类别置信度. [batchsize, 8732, 21]    
        location： 预测坐标偏移     [batchsize, 8732, 4]
        prior_boxes: prior_boxes  [8732,4] [cx, cy, w, h]
        labels:  实际标签.  [batchsize, n, 5] 5 means[xmin, ymin, xmax, ymax, class_id]
        """
        batchsize = confidence.shape[0]
        prior_box_num = confidence.shape[1]
        label_num = labels.shape[1]
        # print(confidence.device)
        # print(location.device)
        # print(prior_boxes.device)
        # print(labels.device)
        # b
        print(prior_boxes.shape)
        prior_boxes = prior_boxes.unsqueeze(0)
        print(prior_boxes.shape)
        prior_boxes = prior_boxes.expand(batchsize, prior_box_num, 4)
        print(prior_boxes.shape)

        loc_t = torch.Tensor(num, num_priors, 4)
        # prior_boxes_matched = matchPriorBox(prior_boxes, labels)
        #[batchsize, 8732, 5] [cx, cy, w, h, class_id]
        # print(prior_boxes_matched.shape)
        # print(prior_boxes_matched.squeenze().shape)
        b



        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
                
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

        # print(labels.view(-1))
        # print(self.alpha)
        # self.alpha = self.alpha.gather(0,labels.view(-1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        #loss = torch.mul(self.alpha, loss.t())
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


