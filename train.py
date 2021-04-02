import torch
import torch.nn as nn
import numpy as np
import cv2
import random

import torch.optim as optim


from libs.backbone import BackboneVGG16, SSD
from libs.data import getDataLoader
from libs.utils import getAllName, seedReproducer
from libs.loss import myLoss
from libs.box_utils import generateProirBox


import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def getSchedu(schedu, optimizer):
    if schedu=='default':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1)
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in schedu: 
        T_0 = int(schedu.strip().split('-')[1])
        T_mult = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=T_0, 
                                                            T_mult=T_mult)
    return scheduler


def getOptimizer(optims, model, learning_rate, weight_decay):
    if optims=='adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optims=='AdaBelief':
        optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-12, betas=(0.9,0.999))
    elif optims=='Ranger':
        optimizer = Ranger(model.parameters(), lr=learning_rate)
    return optimizer



def train(data_loader, epoch, total_epoch, model, criterion, device):
    # switch to evaluate mode
    model.train()

    for batch_idx, (img, labels, file_name) in enumerate(data_loader):
        # print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

        #print(file_name)
        img = img.to(device)
        labels = labels.to(device)

        confidence, location = model(img)

        prior_boxes = generateProirBox()
        prior_boxes = torch.from_numpy(prior_boxes).to(device) #[1, 8732, 4] [cx, cy, w, h]
        loss = criterion(confidence, location, prior_boxes, labels)


        loss.backward() #计算梯度

        optimizer.step() #更新参数
        optimizer.zero_grad()#把梯度置零

        # batch_pred_score = pred_score.data.cpu().numpy()#.tolist()
        # print(batch_pred_score.shape)
        # print(np.max(batch_pred_score[0]), np.argmax(batch_pred_score[0]))

        if batch_idx % 10 == 0:
            print('\r',
                '{}/{} [{}/{} ({:.0f}%)] -  loss: {:.4f} '.format(
                epoch+1, total_epoch, batch_idx * len(img), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), 
                loss.item(),train_acc), 
                end="",flush=True)
            # ETA   it/s


    return 1


def getTrainValNames(name_file, split_ratio):

    with open(name_file, 'r') as f:
        names = f.readlines()
    print(len(names))
    names = [x.strip() for x in names]
    # print(names[:3])
    random.shuffle(names)
    # print(names[:3])

    train_count = int(len(names)*(1-split_ratio))
    
    train_names = names[:train_count]
    val_names = names[train_count:]
    
    return train_names, val_names


if __name__ == '__main__':

    ### 1.config
    random_seed = 42
    seedReproducer(random_seed)

    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    batchsize = 1
    epochs = 30

    ### 2.data
    voc_dir = "../data/VOC2007/trainval/"
    name_file = os.path.join(voc_dir, "ImageSets/Main/trainval.txt")
    train_names,val_names = getTrainValNames(name_file, split_ratio = 0.05)
    train_loader = getDataLoader("train", voc_dir, train_names, 300, batchsize, kwargs)
    print("len train_loader: ", len(train_loader))

    val_loader = getDataLoader("val", voc_dir, val_names, 300, 1, kwargs)
    print("len val_loader: ", len(val_loader))


    ### 3.model
    
    classes = 20
    pretrained_path = "./data/models/vgg16-397923af.pth"
    model = SSD(classes,  pretrained_path).to(device)
    #print(model)
    #b
    learning_rate = 0.01
    weight_decay = 0.0001
    optimizer = getOptimizer('adam', model, learning_rate, weight_decay)
    scheduler = getSchedu('default', optimizer)
    criterion = myLoss().cuda()

    ### 4. train
    for epoch in range(epochs):

        train(train_loader, epoch, epochs, model, criterion, device)

        scheduler.step()


    del model
    gc.collect()
    torch.cuda.empty_cache()
