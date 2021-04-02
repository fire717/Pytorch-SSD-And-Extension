
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform

import xml.etree.ElementTree as xmlET

CLASS_NAME_LIST = ['__background__', # always index 0
                    'aeroplane',
                    'bicycle', 
                    'bird', 
                    'boat', 
                    'bottle', 
                    'bus', 
                    'car', 
                    'cat', 
                    'chair', 
                    'cow', 
                    'diningtable', 
                    'dog', 
                    'horse',
                    'motorbike', 
                    'person', 
                    'pottedplant', 
                    'sheep', 
                    'sofa', 
                    'train', 
                    'tvmonitor']

def readXML(xml_path):
    # return ndarry: [[x0,y0,x1,y1,class],..]
    # value:0-1
    labels = []


    tree = xmlET.parse(xml_path)

    size = tree.findall('size')  
    w = float(size[0].find('width').text)
    h = float(size[0].find('height').text)

    objs = tree.findall('object')    
    for ix, obj in enumerate(objs):
        tmp = []
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)/w
        ymin = float(bbox.find('ymin').text)/h
        xmax = float(bbox.find('xmax').text)/w
        ymax = float(bbox.find('ymax').text)/h
        
        cla = obj.find('name').text 
        if cla not in CLASS_NAME_LIST:
            raise Exception("class name not in CLASS_NAME_LIST")

        labels.append(np.array([xmin, ymin, xmax, ymax, CLASS_NAME_LIST.index(cla)]))

    return labels

###### 1.Data aug
class TrainDataAug:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # raw_h, raw_w = img.shape[:2]
        # min_size = max(img.shape[:2])


        # input_h, input_w = img.shape[:2]
        # h_ratio = input_h/self.h
        # w_ratio = input_w/self.w
        # if (h_ratio>1 and w_ratio>1):
        #     if h_ratio>w_ratio:
        #         resize_h = self.h
        #         resize_w = int(input_w/h_ratio)
        #     else:
        #         resize_w = self.w
        #         resize_h = int(input_h/w_ratio)
        #     img = A.Resize(resize_h,resize_w,cv2.INTER_AREA)(image=img)['image']
        #     min_size = max(img.shape[:2])

        # img = A.OneOf([A.PadIfNeeded(min_height=min_size, min_width=min_size, 
        #                 border_mode=3, value=0, mask_value=0, 
        #                 always_apply=False, p=0.7),
        #             A.PadIfNeeded(min_height=min_size, min_width=min_size, 
        #                 border_mode=0, value=0, mask_value=0, 
        #                 always_apply=False, p=0.3)],
        #                 p=1.0)(image=img)['image']
        

        # img = A.ShiftScaleRotate(
        #                         shift_limit=0.1,
        #                         scale_limit=0.1,
        #                         rotate_limit=20,
        #                         interpolation=cv2.INTER_LINEAR,
        #                         border_mode=cv2.BORDER_CONSTANT,
        #                          value=0, mask_value=0,
        #                         p=0.6)(image=img)['image']

        # img = A.HorizontalFlip(p=0.5)(image=img)['image'] 
        
        # img = A.OneOf([A.RandomBrightness(limit=0.1, p=1), 
        #             A.RandomContrast(limit=0.1, p=1),
        #             A.RandomGamma(gamma_limit=(50, 150),p=1),
        #             A.HueSaturationValue(hue_shift_limit=10, 
        #                 sat_shift_limit=10, val_shift_limit=10,  p=1)], 
        #             p=0.8)(image=img)['image']

        
        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        # img = A.OneOf([A.MotionBlur(blur_limit=3, p=0.2), 
        #                 A.MedianBlur(blur_limit=3, p=0.2), 
        #                 A.GaussianBlur(blur_limit=3, p=0.1),
        #                 A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5)], 
        #                 p=0.8)(image=img)['image']


        
        img = Image.fromarray(img)
        return img


class ValTestDataAug:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        img = Image.fromarray(img)
        return img



######## 2.dataloader

class TensorDatasetTestBackbone(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # img = cv2.imread(self.train_jpg[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,(180, 180))

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])
        #img = imgPaddingWrap(img)
        #b
        if self.transform is not None:
            img = self.transform(img)

        return img, self.train_jpg[index]

    def __len__(self):
        return len(self.train_jpg)






class DatasetTrainVal(Dataset):

    def __init__(self, voc_dir, data_list, transform=None):
        self.voc_dir = voc_dir
        self.data_list = data_list
        self.transform = transform


    def __getitem__(self, index):


        name = self.data_list[index]
        img_path = os.path.join(self.voc_dir, 'JPEGImages', name+".jpg")
        img = cv2.imread(img_path)

        img = self.transform(img)

        xml_path = os.path.join(self.voc_dir, 'Annotations', name+'.xml')

        labels = readXML(xml_path)

        # print(img.shape, torch.from_numpy(np.array(labels)).shape)
        # bb
        return img, torch.from_numpy(np.array(labels)), name
        
    def __len__(self):
        return len(self.data_list)



class DatasetTest(Dataset):

    def __init__(self, train_jpg, transform=None, distill=False):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
        self.distill = distill
        if distill:
            distill_path = r"save/good/result_distill.json"
            print("distill path: ", distill_path)
            with open(distill_path,'r') as f:
                self.distill_dict = json.loads(f.readlines()[0])  

    def __getitem__(self, index):

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])

        if self.transform is not None:
            img = self.transform(img)

        # y = np.array([0,0,0], dtype=np.float32)
        #print(self.train_jpg[index])
        # if 'calling_images' in self.train_jpg[index]:
        y = 0
        if  'smok' in self.train_jpg[index] and 'call' in self.train_jpg[index]:
            y = 3
        elif  'normal' in self.train_jpg[index]:
            y = 1
        elif  'smok' in self.train_jpg[index]:
            y = 2
        # print(y)
        # b
        if self.distill:
            y_onehot = [0,0,0,0]
            y_onehot[y] = 1
            y_onehot = np.array(y_onehot)
            if os.path.basename(self.train_jpg[index]) in self.distill_dict:
                y = y_onehot*0.6+np.array(self.distill_dict[os.path.basename(self.train_jpg[index])])*0.4
            else:
                y = y_onehot*0.9 + (1-0.9)/4
        return img, y
        
    def __len__(self):
        return len(self.train_jpg)



###### 3. get data loader 


def getDataLoader(mode, voc_dir, data_list, img_size, batch_size, kwargs):
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 4



    if mode=="train":
        
        train_loader = torch.utils.data.DataLoader(DatasetTrainVal(voc_dir,
                                                        data_list,
                                                        T.Compose([
                                                            TrainDataAug(img_size, img_size),
                                                            T.ToTensor(),
                                                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
                                                    ),
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    **kwargs
                                                    )


        return train_loader

    elif mode=="val":
        val_loader = torch.utils.data.DataLoader(DatasetTrainVal(voc_dir,
                                                        data_list,
                                                        T.Compose([
                                                            ValTestDataAug(img_size, img_size),
                                                            T.ToTensor(),
                                                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
                                                    ), 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=kwargs['num_workers'], 
                                                    pin_memory=kwargs['pin_memory']
                                                    )

        return val_loader

    elif mode=="test":
        test_loader = torch.utils.data.DataLoader(DatasetTest(voc_dir,
                                                        data_list,
                                                        T.Compose([
                                                            ValTestDataAug(img_size, img_size),
                                                            T.ToTensor(),
                                                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
                                                    ), 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=kwargs['num_workers'], 
                                                    pin_memory=kwargs['pin_memory']
                                                    )

        return test_loader

    elif mode=="testBackbone":
        test_loader = torch.utils.data.DataLoader(
                        TensorDatasetTestBackbone(data_list,
                        T.Compose([
                                    ValTestDataAug(img_size, img_size),
                                    T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                        ), 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=kwargs['num_workers'], 
                        pin_memory=kwargs['pin_memory']
                        )

        return test_loader




