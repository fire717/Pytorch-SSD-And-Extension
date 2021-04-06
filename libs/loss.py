
#!-*- coding: utf-8 -*-

from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable

from libs.box_utils import *


variances = [0.1, 0.2]
iou_threshold = 0.5



def encodeOffset(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4]. [xmin, ymin, xmax, ymax]
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4]. [cx, cy, w, h]
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def boxToOffset(batch_id, labels_box, labels_class, 
                prior_boxes_corner, prior_boxes, obj_count,
                loc_t, conf_t):
    # 把原始的标注框、anchor框等转为对应shape的偏移值，用于后面loss计算
    # labels  [1,n,4]
    # labels_class [1,n,1]
    # prior_boxes_corner [8732,4]  [xmin, ymin, xmax, ymax]
    # output: loc_t, conf_t
    #print(labels_box.shape, labels_class.shape)
    labels_box = labels_box[:obj_count]
    labels_class = labels_class[:obj_count]
    #print(labels_box.shape, labels_class.shape)
    #b
    #print(labels[0].shape, labels[0])
    overlaps = jaccard(labels_box, prior_boxes_corner)
    #[label_num, 8732]
    #print(overlaps.shape, overlaps[0][:10], overlaps[1][:10],)

    # (Bipartite Matching)
    # 每一个gt框找一个最匹配的prior box
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    #print(best_prior_overlap.shape, best_prior_idx.shape) # [2,1]
    #print(best_prior_overlap, best_prior_idx)

    # 每一个prior box找一个最匹配的gt框
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    #print(best_truth_overlap.shape, best_truth_idx.shape) # [1,8732]
    #print(best_truth_overlap, best_truth_idx)

    best_truth_idx.squeeze_(0) #In-place version of squeeze()
    best_truth_overlap.squeeze_(0)
    #[8732,]
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    #[2,]
    # print(best_prior_idx)
    # print(best_truth_overlap[best_prior_idx])
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    #根据idx把对应位置上的iou设为2
    # print(best_truth_overlap[best_prior_idx])

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        #print("1: ", best_truth_idx[best_prior_idx[j]])
        best_truth_idx[best_prior_idx[j]] = j
        #print(best_truth_idx[best_prior_idx[j]])

    #print(labels[batch_id][:, :4].shape) # [2,4]
    #print(best_truth_idx, best_truth_idx.dtype)
    matches = labels_box[best_truth_idx]   
    #best_truth_idx int, [0,0,...,1,1,..] 把 labels[batch_id][:, :4] 0,1两行按列表复制
    #这一步就把标签n*4 变成了8732*4  然后后面再通过iou过滤
    #print(matches.shape)  # Shape: [8732,4]
    #print(matches[:2])
    loc = encodeOffset(matches, prior_boxes, variances)
    # [8732,4] encoded offsets to learn
    loc_t[batch_id] = loc    


    conf = labels_class[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < iou_threshold] = 0  # label as background
    # [8732,] top class label for each prior
    conf_t[batch_id] = conf  


def logSumExp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    防止溢出 https://blog.csdn.net/zziahgf/article/details/78489562
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max




class myLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, negpos_ratio=3, use_gpu=True):
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

        self.num_classes = num_classes
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.negpos_ratio = negpos_ratio

    def forward(self, confidence, location, prior_boxes, labels, obj_count):
        """
        confidence: 预测类别置信度. [batchsize, 8732, 21]    
        location: 预测坐标偏移     [batchsize, 8732, 4]
        prior_boxes: prior_boxes  [8732,4] [cx, cy, w, h]
        labels:  实际标签.  [batchsize, n, 5] 5 means[xmin, ymin, xmax, ymax, class_id]
                            对于n>1，n取值为一个batch中n最多的数量
        obj_count: pad后的labels中实际由多少个框， [batchsize, 1]
        """
        batchsize = confidence.shape[0]
        prior_box_num = confidence.shape[1]
        # label_num = labels.shape[1]
        # print(confidence.device)
        # print(location.device)
        # print(prior_boxes.device)
        # print(labels.device)
        # b
        

        #print("prior_boxes_corner1: ", prior_boxes.shape, prior_boxes[0])
        prior_boxes_corner = pointMidToCorner(prior_boxes)
        # print("prior_boxes_corner2: ", prior_boxes_corner.shape, prior_boxes_corner[0])
        # b

        loc_t = torch.Tensor(batchsize, prior_box_num, 4)
        conf_t = torch.LongTensor(batchsize, prior_box_num)
        # print(loc_t[0][0]) 
        for batch_id in range(batchsize):
            boxToOffset(batch_id, 
                        labels[batch_id][:, :4],
                        labels[batch_id][:, 4],
                        prior_boxes_corner,
                        prior_boxes,
                        obj_count[batch_id],
                        loc_t,
                        conf_t
                        )

        # print(loc_t[0][0]) 
        # b
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 #[batchsize, 8732]
        num_pos = pos.sum(dim=1, keepdim=True)
        #print(num_pos.shape) [batchsize, 1]
        #b

        ### Localization Loss (Smooth L1)
        # 取非背景类的对应坐标
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(location) # [batch,num_priors,4]
        loc_pos = location[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(loc_pos, loc_t, size_average=False)


        ### confidence loss CE
        # Compute max conf across batch for hard negative mining
        batch_conf = confidence.view(-1, self.num_classes) # (n*8732)*21
        # print('\n------------------')
        # print(batch_conf.shape)
        # print(conf_t.view(-1, 1).shape)
        # for i in range(80):
        #     print(conf_t[0][i*100:(i+1)*100])
        # print(batch_conf.gather(1, conf_t.view(-1, 1)).shape) # (n*8732)*1
        # print('------------------\n')
        loss_conf = logSumExp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # (n*8732)*1
        #gather: 把batch_conf按conf_t的形式在维度1聚合
        #这里batch_conf.gather(1, conf_t.view(-1, 1))其实就是batch_conf对应index的类别的值


        # Hard Negative Mining  先把正样本置0 然后剩下的降序排序 取前面的
        loss_conf = loss_conf.view(batchsize, -1) #n*(8732*1)
        # print(loss_conf)
        # print(pos)
        loss_conf[pos] = 0  # filter out pos boxes for now

        #print(loss_conf.shape) #batchsize*8732
        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1) 
        #升序 排序结果索引再排序的索引 因为升序又相当于把索引还原，
        #对应的排序索引就是原始数组每一个元素的降序排序序列号
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        #print(num_neg.shape, idx_rank.shape)#[batchsize, 1] [batchsize, 8732]
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # print(num_neg.expand_as(idx_rank)[0][:20])
        # print(idx_rank[0][:20])
        #print(neg.shape) #[batchsize, 8732] 前top个的索引id为true，其余为false
        # b
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(confidence)
        #[batchsize, 8732]->[batchsize, 8732,21]
        neg_idx = neg.unsqueeze(2).expand_as(confidence)
        conf_p = confidence[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        #[batchsize, (pos_idx+neg_idx),21] -> [batchsize*(pos_idx+neg_idx),21]
        # gt(0) 逐元素比较 大于0的为True 反之False
        targets_weighted = conf_t[(pos+neg).gt(0)] #anchor box中与标签iou满足重叠条件的正负样本框类别
        #[batchsize*(pos_idx+neg_idx),]
        loss_conf = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        #F.cross_entropy(out, y) out没有经过softmax， y是标量


        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        if N==0:
            loss_loc = loss_conf = 0
        else:
            loss_loc = self.alpha*loss_loc/N
            loss_conf /= N
        return loss_loc, loss_conf

