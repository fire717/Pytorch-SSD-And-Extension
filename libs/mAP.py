#https://github.com/rbgirshick/py-faster-rcnn/blob/781a917b378dbfdedb45b6a56189a31982da1b43/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

# transfer from py2 to py3

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:  #VOC在2010之后换了评价方法，所以决定是否用07年的
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):  #  07年的采用11个点平分recall来计算
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 取一个recall阈值之后最大的precision
            ap = ap + p / 11.  # 将11个precision加和平均
    else:  # 这里是用2010年后的方法，取所有不同的recall对应的点处的精度值做平均，不再是固定的11个点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  #recall和precision前后分别加了一个值，因为recall最后是1，所以
        mpre = np.concatenate(([0.], prec, [0.])) # 右边加了1，precision加的是0

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #从后往前，排除之前局部增加的precison情况

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 这里巧妙的错位，返回刚好TP的位置，
                                                                                      # 可以看后面辅助的例子

        # and sum (\Delta recall) * prec   用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    
# 计算每个类别对应的AP，mAP是所有类别AP的平均值
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print(imagenames)
    # b
    if not os.path.isfile(cachefile):
        # load annots 
        # 这里提取的是所有测试图片中的所有object gt信息, 07年的test真实标注是可获得的，12年就没有了
        recs = {}
        for i, imagename in enumerate(imagenames):
            # print('--')
            # print(imagename)
            # print(annopath % (imagename))
            # b
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            
            try:
                recs = pickle.load(f)
            except EOFError:
                print("cachefile is empty.")
                return 0,0,0

    # extract gt objects for this class 从上面的recs提取我们要判断的那类标注信息
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) # 该图片中该类别对应的所有bbox的是否已被匹配的标志位
        npos = npos + sum(~difficult) #累计所有图片中的该类别目标的总数，不算diffcult
                                                                     # 这里计算还是很巧妙的，npos=TP+FN
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    # 读取相应类别的检测结果文件，每一行对应一个检测目标
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    if BB.shape[0]==0:
        print("BB is empty.")
        return 0,0,0

    # sort by confidence 按置信度由大到小排序
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # 检测结果文件的行数
    tp = np.zeros(nd) # 用于标记每个检测结果是tp还是fp
    fp = np.zeros(nd)
    for d in range(nd):
       # 取出该条检测结果所属图片中的所有ground truth
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps  计算与该图片中所有ground truth的最大重叠度
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        # 这里就是具体的分配TP和FP的规则了
        if ovmax > ovthresh:  # 如果最大的重叠度大于一定的阈值
            if not R['difficult'][jmax]: # 如果最大重叠度对应的ground truth为difficult就忽略，
                                                               # 因为上面npos就没算
                if not R['det'][jmax]: # 如果对应的最大重叠度的ground truth以前没被匹配过则匹配成功，即tp
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:  # 若之前有置信度更高的检测结果匹配过这个ground truth，则此次检测结果为fp
                    fp[d] = 1.
        else:
            # 该图片中没有对应类别的目标ground truth或者与所有ground truth重叠度都小于阈值
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp) # 累加函数np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
