import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import random

import torch.optim as optim
# import gc
#import torch.nn.functional as F

from libs.backbone import BackboneVGG16, SSD
from libs.data import getDataLoader, CLASS_NAME_LIST
from libs.utils import getAllName, seedReproducer
from libs.loss import myLoss
from libs.box_utils import generateProirBox
from libs.ssd import decodeOutput

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'





def test(data_loader, model, device, save_dir, test_count):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, labels, obj_count, file_name,w,h) in enumerate(data_loader):
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

    data_loader = getDataLoader("eval", voc_dir, data_sets, 300, batchsize, kwargs)
    print("len data_loader: ", len(data_loader))



    ### 3.model
    
    classes = 20
    model = SSD(classes).to(device)
    model_path = "data/save/ssd_e152_1.45554.pth"
    model.load_state_dict(torch.load(model_path))


    ### 4. test
    save_dir = 'data/output'
    test_count = 20
    test(data_loader, model, device, save_dir, test_count)



    # del model
    # gc.collect()
    # torch.cuda.empty_cache()



