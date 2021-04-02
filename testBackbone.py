import torch
import torch.nn as nn
import numpy as np
import cv2


from libs.backbone import BackboneVGG16, SSD
from libs.data import getDataLoader
from libs.utils import getAllName, seedReproducer

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'






def predict(test_loader, model, device):
    # switch to evaluate mode
    model.eval()

    res_list = []
    with torch.no_grad():
        #end = time.time()
        pres = []
        labels = []
        for i, (data, img_name) in enumerate(test_loader):
            # print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

            print(img_name)
            data = data.to(device)
            output = model(data)

            print(output[0][:10])
            pred_score = nn.Softmax(dim=1)(output)
            print(pred_score[0][:10])
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()


            batch_pred_score = pred_score.data.cpu().numpy()#.tolist()
            print(batch_pred_score.shape)
            print(np.max(batch_pred_score[0]), np.argmax(batch_pred_score[0]))

    return 1



if __name__ == "__main__":

    random_seed = 42
    seedReproducer(random_seed)


    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    classes = 20
    pretrained_path = "./data/models/vgg16-397923af.pth"
    model = SSD(classes,  pretrained_path).to(device)
    print(model)

    voc_dir = "../data/VOC2007/trainval/"
    img_path = "./data/test"
    img_names = getAllName(img_path)

    test_loader = getDataLoader("testBackbone", voc_dir, img_names, 300, 1, kwargs)
    print("len test_loader: ", len(test_loader))


    predict(test_loader, model, device)
