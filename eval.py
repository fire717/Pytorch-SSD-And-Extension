import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import random
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import gc
#import torch.nn.functional as F

from libs.backbone import BackboneVGG16, SSD
from libs.data import getDataLoader, CLASS_NAME_LIST
from libs.utils import getAllName, seedReproducer
from libs.loss import myLoss
from libs.box_utils import generateProirBox
from libs.ssd import decodeOutput
from libs.mAP import voc_eval

import os



os.environ["CUDA_VISIBLE_DEVICES"] = '0'



VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')




def eval(data_loader, model, device, voc_dir, save_dir):
    # switch to evaluate mode
    model.eval()

    

    num_images = len(data_loader)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(VOC_CLASSES)+1)]

    data_basename = []
    with torch.no_grad():
        for batch_idx, (img, labels, obj_count, file_name,w,h) in enumerate(data_loader):
            #if(batch_idx>=100):break
                

            # trained_items += img.shape[0]#batchsize、
            #print(file_name)
            img = img.to(device)
            labels = labels.to(device)


            #print("train: --- ", img.shape, labels.shape)
            t = time.time()
            confidence, location = model(img)
            t1 = time.time() - t
            prior_boxes = generateProirBox()
            prior_boxes = torch.from_numpy(prior_boxes).to(device)

            results = decodeOutput(confidence, location, prior_boxes)
            t2 = time.time() - t
            for i,result in enumerate(results):
                # print(result)
                # b
                result[0] *= w
                result[1] *= h
                result[2] *= w
                result[3] *= h
                all_boxes[i][batch_idx] = np.array([result[:5]])

            # show_img = cv2.imread(file_name[0])
            # h,w = show_img.shape[:2]
            # for box in results:
            #     x0 = int(box[0]*w)
            #     y0 = int(box[1]*h)
            #     x1 = int(box[2]*w)
            #     y1 = int(box[3]*h)
            #     cv2.rectangle(show_img,(x0,y0),(x1,y1),(255,0,0),1)

            #cv2.imwrite(os.path.join(save_dir, os.path.basename(file_name[0])), show_img)
            data_basename.append(os.path.basename(file_name[0]).split('.')[0])

            print_line = '{}/{} infer time: {:.4f} total time: {:.4f}'.format(
                batch_idx, num_images,t1,t2)
            print("\r",print_line,end="",flush=True)

    #all_boxes = np.array(all_boxes)
    # print(len(data_basename), data_basename[0])
    # b
    evaluate_detections(all_boxes, voc_dir, save_dir, data_loader,data_basename)


def evaluate_detections(box_list, voc_dir, output_dir, dataset,data_basename):
    write_voc_results_file(box_list, dataset, voc_dir,data_basename)
    do_python_eval(voc_dir, output_dir)




def get_voc_results_file_template(voc_dir,image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(voc_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset, voc_dir,data_basename):
    for cls_ind, cls in enumerate(VOC_CLASSES):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(voc_dir,'test', cls)
        #print(len(dataset))
        with open(filename, 'wt') as f:
            for im_ind, basename in enumerate(data_basename):
                #0 ('/home/AlgorithmicGroup/yw/workshop/mine/data/VOCdevkit/VOC2007', '000001')
                print("\r",im_ind,end="",flush=True)
               
                # print(basename)
                # b
                dets = all_boxes[cls_ind+1][im_ind]
                # print(all_boxes.shape)
                #print("dets 0:",dets)
                if len(dets) == 0:
                    continue
                #print("dets 1:",dets)
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(basename, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(voc_dir,output_dir='output', use_07=True):
    cachedir = os.path.join(voc_dir, 'annotations_cache')
    aps = []

    annopath = os.path.join(voc_dir, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(voc_dir, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(voc_dir, 'VOC2007', 'ImageSets',
                              'Main', '{:s}.txt')
    #print(imgsetpath.format('test'))

    #b
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(VOC_CLASSES):
        filename = get_voc_results_file_template(voc_dir, 'test', cls)
        print("filename: ", filename)
        # b
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format('test'), cls, cachedir,
           ovthresh=0.5, use_07_metric=True)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))


if __name__ == '__main__':

    ### 1.config
    random_seed = 42
    seedReproducer(random_seed)

    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    cudnn.deterministic = False
    
    batchsize = 1



    ### 2.data
    voc_dir = "../data/VOCdevkit/"
    data_sets = ['VOC2007']

    data_loader = getDataLoader("eval", voc_dir, data_sets, 300, batchsize, kwargs)
    print("len data_loader: ", len(data_loader))



    ### 3.model
    
    classes = 20
    model = SSD(classes).to(device)
    model_path = "data/save/ssd_e152_1.45554.pth"
    model.load_state_dict(torch.load(model_path))


    ### 4. test
    save_dir = 'data/output/mAP'
    #test_count = 10
    eval(data_loader, model, device, voc_dir, save_dir)



    # del model
    # gc.collect()
    # torch.cuda.empty_cache()



