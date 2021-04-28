import torch
import torch.nn.functional as F

from libs.data import CLASS_NAME_LIST
from libs.box_utils import nms

def decodeOutput(confidence, location, prior_boxes,
                    top_k=200, conf_thresh=0.01, nms_thresh=0.45,
                    score_thresh=0.6,variances = [0.1, 0.2]):

    # return [[x0,y0,x1,y1,score,class],...]

    
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