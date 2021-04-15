
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

CLASS_NAME_LIST = [#'__background__', # always index 0
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

    def __init__(self, data_type, voc_dir, data_sets, transform=None):
        self.data_type = data_type
        self.voc_dir = voc_dir
        self.data_sets = data_sets
        self.transform = transform

        self._getDataList()


    def _getDataList(self):
        self.img_list = []
        self.xml_list = []
        if self.data_type == 'trainval':
            name_file = 'trainval_part.txt'
            for data_set in self.data_sets:
                base_dir = os.path.join(self.voc_dir, data_set)
                name_path = os.path.join(base_dir, "ImageSets","Main",name_file)
                with open(name_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    name = line.strip()
                    img_path = os.path.join(base_dir, 'JPEGImages', name+".jpg")
                    xml_path = os.path.join(base_dir, 'Annotations', name+'.xml')
                    self.img_list.append(img_path)
                    self.xml_list.append(xml_path)


        elif self.data_type == 'train':
            name_file = 'train.txt'
            for data_set in self.data_sets:
                base_dir = os.path.join(self.voc_dir, data_set)
                name_path = os.path.join(base_dir, "ImageSets","Main",name_file)
                with open(name_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    name = line.strip()
                    img_path = os.path.join(base_dir, 'JPEGImages', name+".jpg")
                    xml_path = os.path.join(base_dir, 'Annotations', name+'.xml')
                    self.img_list.append(img_path)
                    self.xml_list.append(xml_path) 

        elif self.data_type == 'val':
            name_file = 'val.txt'
            for data_set in self.data_sets:
                base_dir = os.path.join(self.voc_dir, data_set)
                name_path = os.path.join(base_dir, "ImageSets","Main",name_file)
                with open(name_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    name = line.strip()
                    img_path = os.path.join(base_dir, 'JPEGImages', name+".jpg")
                    xml_path = os.path.join(base_dir, 'Annotations', name+'.xml')
                    self.img_list.append(img_path)
                    self.xml_list.append(xml_path)           


    def __getitem__(self, index):


        img_path = self.img_list[index]
        # img_path = os.path.join(self.voc_dir, 'JPEGImages', name+".jpg")
        # print(img_path)
        img = cv2.imread(img_path)

        if self.transform:
            img = self.transform(img)

        # xml_path = os.path.join(self.voc_dir, 'Annotations', name+'.xml')
        xml_path = self.xml_list[index]
        # print(xml_path)
        # b
        labels = readXML(xml_path)

        #print(np.array(labels).shape)
        # print(img.shape, torch.from_numpy(np.array(labels)).shape)
        #bb
        return img, torch.from_numpy(np.array(labels)), len(labels), img_path
        
    def __len__(self):
        return len(self.img_list)



class DatasetTest(Dataset):

    def __init__(self, data_type, voc_dir, data_sets, transform=None):
        self.data_type = data_type
        self.voc_dir = voc_dir
        self.data_sets = data_sets
        self.transform = transform

        self._getDataList()


    def _getDataList(self):
        self.img_list = []
        self.xml_list = []
        if self.data_type == 'trainval':
            name_file = 'trainval.txt'
            for data_set in self.data_sets:
                base_dir = os.path.join(self.voc_dir, data_set)
                name_path = os.path.join(base_dir, "ImageSets","Main",name_file)
                with open(name_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    name = line.strip()
                    img_path = os.path.join(base_dir, 'JPEGImages', name+".jpg")
                    xml_path = os.path.join(base_dir, 'Annotations', name+'.xml')
                    self.img_list.append(img_path)
                    self.xml_list.append(xml_path)

        #            


    def __getitem__(self, index):


        img_path = self.img_list[index]
        # img_path = os.path.join(self.voc_dir, 'JPEGImages', name+".jpg")
        # print(img_path)
        img = cv2.imread(img_path)

        if self.transform:
            img = self.transform(img)

        # xml_path = os.path.join(self.voc_dir, 'Annotations', name+'.xml')
        xml_path = self.xml_list[index]
        # print(xml_path)
        # b
        labels = readXML(xml_path)

        #print(np.array(labels).shape)
        # print(img.shape, torch.from_numpy(np.array(labels)).shape)
        #bb
        return img, torch.from_numpy(np.array(labels)), len(labels), img_path
        
    def __len__(self):
        return len(self.img_list)



class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img):
        # image = image.astype(np.float32)
        img = np.array(img, dtype=np.float32) - self.mean
        img = torch.from_numpy(img)

        img = img.permute(2,0,1) #hwc->chw

        return img


###### 3. get data loader 

def collate_fn(batch_data):
    """
    默认的batch合并函数是直接把不同item直接拼接
    但是目标检测中的label，不同图片的目标框数量不同
    直接使用就会报错 stack expects each tensor to be equal size, 
    but got [2, 5] at entry 0 and [11, 5] at entry 1

    所以需要直接实现合并的函数
    """ 
    # print(len(batch_data)) #=batchsize
    # print(len(batch_data[0])) #= len(return thing)  here is 3 (img, torch.from_numpy(np.array(labels)), name)
    # print(batch_data[0][0].shape)  # [3,300,300]
    # print(batch_data[0][1].shape)  # [n_obj, 5]
    # print(batch_data[0][2])        #name

    # max_obj = max([x[1].shape[0] for x in batch_data])
    # print(batch_data[0][1].shape, batch_data[1][1].shape, max_obj)

    def _padBatchBox(box_tensor_list):
        max_obj = max([x.shape[0] for x in box_tensor_list])

        # print(box_tensor_list[0].shape, box_tensor_list[1].shape)
        box_tensor_list_with_same_shape = [torch.cat((x, torch.Tensor([[0]*x.shape[1]]*(max_obj-x.shape[0])))) for x in box_tensor_list]
        # print(box_tensor_list_with_same_shape[0].shape, box_tensor_list_with_same_shape[1].shape)
        # b
        return box_tensor_list_with_same_shape


    batch_img = torch.stack([x[0] for x in batch_data], 0)
    batch_box = torch.stack(_padBatchBox([x[1] for x in batch_data]), 0)
    batch_len_obj = [x[2] for x in batch_data]
    batch_name = [x[3] for x in batch_data]
    return batch_img, batch_box, batch_len_obj, batch_name


def getDataLoader(mode, voc_dir, data_sets, img_size, batch_size, kwargs):
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 4


    if mode=="train":
        
        train_loader = torch.utils.data.DataLoader(DatasetTrainVal("trainval",
                                                        voc_dir,
                                                        data_sets,
                                                        T.Compose([
                                                            TrainDataAug(img_size, img_size),
                                                            # SubtractMeans([123, 117, 104])#bgr
                                                            T.ToTensor(),
                                                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
                                                    ),
                                                    collate_fn = collate_fn,
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    **kwargs
                                                    )

        return train_loader

    elif mode=="trainval":
        
        train_loader = torch.utils.data.DataLoader(DatasetTrainVal("train",
                                                        voc_dir,
                                                        data_sets,
                                                        T.Compose([
                                                            TrainDataAug(img_size, img_size),
                                                            T.ToTensor(),
                                                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
                                                    ),
                                                    collate_fn = collate_fn,
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    **kwargs
                                                    )

        val_loader = torch.utils.data.DataLoader(DatasetTrainVal("val",
                                                        voc_dir,
                                                        data_sets,
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

        return train_loader,val_loader

    elif mode=="val":
        val_loader = torch.utils.data.DataLoader(DatasetTrainVal("val",
                                                        voc_dir,
                                                        data_sets,
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
        test_loader = torch.utils.data.DataLoader(DatasetTest("test",
                                                        voc_dir,
                                                        data_sets,
                                                        T.Compose([
                                                            ValTestDataAug(img_size, img_size),
                                                            T.ToTensor(),
                                                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                        ])
                                                    ), 
                                                    batch_size=1, 
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




