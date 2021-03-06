import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import time

import torch.optim as optim
import torch.backends.cudnn as cudnn
# import gc

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
        gamma = float(schedu.strip().split('-')[2])
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
    # print(model)
    # b
    # loc_loss = 0
    # conf_loss = 0
    # trained_items = 0
    #print(len(data_loader.dataset), len(data_loader))#5011 209
    #b
    t = time.time()
    for batch_idx, (img, labels, obj_count, file_name) in enumerate(data_loader):
        # print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)
        # trained_items += img.shape[0]#batchsize???
        #print(file_name)
        # print(batch_idx ,img.shape[0])
        # b
        t0 = time.time()
        img = img.to(device)
        labels = labels.to(device)

        t1 = time.time()
        #print("train: --- ", img.shape, labels.shape)
        confidence, location = model(img)
        t2 = time.time()

        prior_boxes = generateProirBox()
        prior_boxes = torch.from_numpy(prior_boxes).to(device) #[1, 8732, 4] [cx, cy, w, h]
        loss_l, loss_c = criterion(confidence, location, prior_boxes, labels, obj_count)
        loss = loss_l + loss_c

        loss.backward() #????????????

        optimizer.step() #????????????
        optimizer.zero_grad()#???????????????
        t3 = time.time()
        # batch_pred_score = pred_score.data.cpu().numpy()#.tolist()
        # print(batch_pred_score.shape)
        # print(np.max(batch_pred_score[0]), np.argmax(batch_pred_score[0]))
        #print(loss_l, loss_c)
        # print(loss_l.item())
        # loc_loss += loss_l.item()
        # print(loss_l.item())
        # print("----")
        # conf_loss += loss_c.item()
        total_loss = loss_l.item()+loss_c.item()

        if batch_idx % 1 == 0:
            print('\r',
                '{}/{} [{}/{} ({:.0f}%)] - loss: {:.4f}  loss_loc: {:.4f} loss_conf: {:.4f} '.format(
                epoch+1, total_epoch, 
                batch_idx * img.shape[0], len(data_loader.dataset),
                100. * batch_idx / len(data_loader), 
                total_loss,
                loss_l.item(),
                loss_c.item()), 
                end="",flush=True)
            # ETA   it/s
        print(t0-t, t1-t0, t2-t1,t3-t2)
        #0.000124 0.00208 0.8137 0.09182
        t = time.time()

    return total_loss




if __name__ == '__main__':

    ### 1.config
    random_seed = 42
    seedReproducer(random_seed)

    device_name = "cuda"
    device = torch.device(device_name)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    batchsize = 24
    epochs = 30

    ### 2.data
    voc_dir = "../data/VOCdevkit/"
    data_sets = ['VOC2007']

    # name_file = []
    # for data_set in data_sets:
    #     name_file.extend([os.path.join(voc_dir, data_set, "ImageSets/Main/trainval.txt")])
    #print(name_file)
    # train_names,val_names = getTrainValNames(name_file, split_ratio = 0.05)
    # train_loader,val_loader = getDataLoader("trainval", voc_dir, data_sets, 300, batchsize, kwargs)
    # print("len train_loader: ", len(train_loader), len(val_loader))
    train_loader = getDataLoader("train", voc_dir, data_sets, 300, batchsize, kwargs)
    print("len train_loader: ", len(train_loader))

    # val_loader = getDataLoader("val", voc_dir, val_names, 300, 1, kwargs)
    # print("len val_loader: ", len(val_loader))


    ### 3.model
    
    classes = 20
    pretrained_path = "./data/models/vgg16-397923af.pth"
    model = SSD(classes,  pretrained_path).to(device)

    if device_name=='cuda':
        print("Use gpu to train")
        model = torch.nn.DataParallel(model)
        #cudnn.benchmark = True
        cudnn.deterministic = False
    # model_path = "data/save/ssd_e62_1.91127.pth"
    # model.load_state_dict(torch.load(model_path))
    #print(model)
    #b
    learning_rate = 0.001
    weight_decay = 0.0001
    optimizer = getOptimizer('SGD', model, learning_rate, weight_decay)
    scheduler = getSchedu('step-4-0.8', optimizer)
    criterion = myLoss(num_classes=classes+1, use_gpu=True).cuda()

    ### 4. train
    for epoch in range(epochs):

        train_loss = train(train_loader, epoch, epochs, model, criterion, device)

        scheduler.step()

        if epoch%10 == 0:
            save_path = './data/save/ssd_e%d_%.5f.pth' % (epoch,train_loss)
            torch.save(model.state_dict(), save_path)

        print('\n')

    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
