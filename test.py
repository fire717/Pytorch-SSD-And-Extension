import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import random

import torch.optim as optim
# import gc
import torch.nn.functional as F

from libs.backbone import BackboneVGG16, SSD
from libs.data import getDataLoader, CLASS_NAME_LIST
from libs.utils import getAllName, seedReproducer
from libs.loss import myLoss
from libs.box_utils import generateProirBox


import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
variances = [0.1, 0.2]


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
 
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]  #输入boxes的所有框的x1
    y1 = boxes[:, 1]  #输入boxes的所有框的y1
    x2 = boxes[:, 2]  #输入boxes的所有框的x2
    y2 = boxes[:, 3]  #输入boxes的所有框的y2
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order  小->大
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals 取前200个大的
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
 
    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0: #前top_k个框列表中若还有框
        i = idx[-1]  # index of current largest val 取最大的框序号为i
        # keep.append(i)
        keep[count] = i #用keep列表记住取出的框的顺序
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view 框序号列表中去掉当前最大的框序号
        # load bboxes of next highest vals
        xx1 = torch.index_select(x1, 0, idx) #xx1是 idx框中的x1
        yy1 = torch.index_select(y1, 0, idx)#yy1是 idx框中的y1
        xx2 = torch.index_select(x2, 0, idx)#xx2是 idx框中的x2
        yy2 = torch.index_select(y2, 0, idx)#yy1是 idx框中的y2
        # store element-wise max with next highest score
        #print(x1[i])
        xx1 = torch.clamp(xx1, min=x1[i].item())
        yy1 = torch.clamp(yy1, min=y1[i].item())
        xx2 = torch.clamp(xx2, max=x2[i].item())
        yy2 = torch.clamp(yy2, max=y2[i].item())
        # w.reshape_as(xx2)
        # h.reshape_as(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou  计算idx中所有框与当前分类置信度最大框的IOU
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)] #idx中IoU大于overlap的框都去除
    return keep, count#keep是同一类中，每个个体的框的序号，count则是个数


def decodeOutput(confidence, location, prior_boxes,
                    top_k=200, conf_thresh=0.01, nms_thresh=0.45,
                    score_thresh=0.6):


    # print(confidence.shape, confidence[0][0])
    # print(location.shape, location[0][0])

    #### 按置信度排序
    #confidence转类别
    confidence = F.softmax(confidence,dim = 2)
    # print(confidence.shape, confidence[0][0], sum(confidence[0][0]))
    #confidence = torch.argmax(confidence, dim = 2, keepdim=True)
    # print(confidence.shape, confidence[0][0], sum(confidence[0][0]))
    confidence = torch.squeeze(confidence,0)
    location = torch.squeeze(location,0)
    # print(confidence.shape, location.shape)
    # print(confidence.device, location.device, prior_boxes.device)

    #### box decode
    # print(location[:4])
    boxes = torch.cat([location[..., :2] * variances[0] * prior_boxes[..., 2:] + prior_boxes[..., :2],
                        torch.exp(location[..., 2:] * variances[1]) * prior_boxes[..., 2:]
                    ], dim=location.dim() - 1)
    # cx cy w h -> x0 y0 x1 y1
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # print(boxes[:4])
    # b

    # print(boxes.shape)

    # class_boxes = torch.cat([boxes,confidence], dim=-1)
    # print(class_boxes.shape, class_boxes[0])

    ## to do: 优化 先获取索引，然后只处理非0类的
    # index = confidence[:,0]>0
    # print(index.shape)
    # print(index)

    # print(class_boxes[class_boxes[:,4]>0].shape)
    # filter_boxes = class_boxes[class_boxes[:,4]>0]
    # print(filter_boxes)

    #### nms  每个类别分别做nms 返回前topk
    # print(len(CLASS_NAME_LIST), confidence.shape, boxes.shape)
    results = []
    for class_id in range(len(CLASS_NAME_LIST)):
        class_mask = confidence[:,class_id+1].gt(conf_thresh)
        #print(class_mask.shape, class_mask.dtype)
        # b
        class_scores = confidence[:,class_id+1][class_mask]
        # print(class_scores.shape)
        if class_scores.size(0) == 0:
            continue
        # l_mask = class_mask.unsqueeze(1).expand_as(decoded_boxes)
        class_boxes = boxes[class_mask]
        #print(class_boxes.shape, class_scores.shape)
        # idx of highest scoring and non-overlapping boxes per class
        ids, count = nms(class_boxes, class_scores, nms_thresh, top_k)
        #print(ids,count,ids[:count],ids.dtype)
        # output[i, class_id, :count] = torch.cat((class_scores[ids[:count]].unsqueeze(1),
        #                                 boxes[ids[:count]]), 1)
        #print(class_scores, class_boxes)
        #print(class_boxes[ids[:count]].shape)
        #print(class_scores[ids[:count]].shape)
        #print(class_scores[ids].unsqueeze(-1).shape)
        result = torch.cat([class_boxes[ids[:count]], class_scores[ids[:count]].unsqueeze(-1)],dim=1)
        #置信度
        result = result[result[:,4].gt(score_thresh)]
        #print(class_id,result.shape)
        if result.shape[0]>0:
            result = result.data.cpu().numpy().tolist()
            # [x0,y0,x1,y1,score,class]
            results.extend([box+[class_id] for box in result])
    
    # print(filter_boxes[:,1:].shape, filter_boxes[:,0].shape)
    # keep = torchvision.ops.nms(filter_boxes[:,:4], filter_boxes[:,4], iou_threshold=nms_thresh)
    # print(keep.shape, keep)
    # print(filter_boxes[keep].shape)
    # result = filter_boxes[keep]
    # pth

    # my own version


    ### 最后一遍过滤 置信度

    # print(results)
    
    # b
    return results 

def test(data_loader, model, device, save_dir, test_count):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, labels, obj_count, file_name) in enumerate(data_loader):
            if batch_idx>=test_count:
                break
            # print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)
            # trained_items += img.shape[0]#batchsize、
            #print(file_name)
            img = img.to(device)
            labels = labels.to(device)


            #print("train: --- ", img.shape, labels.shape)
            confidence, location = model(img)
            prior_boxes = generateProirBox()
            prior_boxes = torch.from_numpy(prior_boxes).to(device)

            results = decodeOutput(confidence, location, prior_boxes)



            show_img = cv2.imread(file_name[0])
            h,w = show_img.shape[:2]
            for box in results:
                x0 = int(box[0]*w)
                y0 = int(box[1]*h)
                x1 = int(box[2]*w)
                y1 = int(box[3]*h)
                cv2.rectangle(show_img,(x0,y0),(x1,y1),(255,0,0),1)

            cv2.imwrite(os.path.join(save_dir, os.path.basename(file_name[0])), show_img)





if __name__ == '__main__':

    ### 1.config
    random_seed = 42
    seedReproducer(random_seed)

    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}


    batchsize = 1



    ### 2.data
    voc_dir = "../data/VOCdevkit/"
    data_sets = ['VOC2007']

    data_loader = getDataLoader("train", voc_dir, data_sets, 300, batchsize, kwargs)
    print("len data_loader: ", len(data_loader))



    ### 3.model
    
    classes = 20
    model = SSD(classes).to(device)
    model_path = "data/save/1/ssd_e62_1.91127.pth"
    model.load_state_dict(torch.load(model_path))


    ### 4. test
    save_dir = 'data/output'
    test_count = 1
    test(data_loader, model, device, save_dir, test_count)



    # del model
    # gc.collect()
    # torch.cuda.empty_cache()



