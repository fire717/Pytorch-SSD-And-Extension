import torch
import torch.nn as nn
# import torchvision.models as models

# import torchvision.transforms as T

import numpy as np
import cv2
import torch.nn.init as init

from libs.utils import copyStateDict
import collections

class VGG16(nn.Module):
    """
    model from torchvision.models vgg16
    
    batch_norm: origin paper and pretrained model not use,default False
                may get better result while add.


    """

    def __init__(self, num_classes=1000, init_weights=True, batch_norm = False):
        super(VGG16, self).__init__()


        self.num_classes = num_classes
        self.batch_norm = batch_norm

        
        # self.features1,self.features2,self.features3,self.features4,self.features5 = self.buildFeatures()
        self.features = self.buildFeatures()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # if init_weights:
        #     self._initialize_weights()


    # def buildFeatures1(self):
    #     layers = []
    #     in_channels = 3
        
    #     cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #     for v in cfg:
    #         if v == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
    #             if self.batch_norm:
    #                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    #             else:
    #                 layers += [conv2d, nn.ReLU(inplace=True)]
    #             in_channels = v
    #     return nn.Sequential(*layers)

    def buildFeatures(self):
        layers = []


        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))



        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))



        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))


        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        
        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))


        return nn.Sequential(*layers)


    def buildFeatures3(self):
        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        features1 = nn.Sequential(*layers)


        layers = []
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features2 = nn.Sequential(*layers)


        layers = []
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features3 = nn.Sequential(*layers)


        layers = []
        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features4 = nn.Sequential(*layers)

        
        layers = []
        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        features5 = nn.Sequential(*layers)

        return features1, features2, features3, features4, features5



    def forward(self, x):
        x = self.features(x)

        # x = self.features1(x)
        # x = self.features2(x)
        # x = self.features3(x)
        # x = self.features4(x)
        # x = self.features5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)





def newLayerLoadPretrained(model, pretrained_path):
    """
    input: model model with random init


    output: model with pretrained weights
    """

    this_state_dict = model.state_dict()
    pretrained_state_dict = torch.load(pretrained_path) 

    new_state_dict = collections.OrderedDict()
    for k,v in pretrained_state_dict.items():    

        layer_num = int(k.split(".")[1])
        if k in name_list2:
            new_state_dict[name_list1[name_list2.index(k)]] = v
        else:
            new_state_dict[k] = v


    this_state_dict.update(new_state_dict)
    model.load_state_dict(this_state_dict, strict=True)

    return model


class BackboneVGG16(nn.Module):
    def __init__(self, classes, pretrained_path):
        super(BackboneVGG16, self).__init__()

        
        self.vgg16 = VGG16()

        self.vgg16 = newLayerLoadPretrained(self.vgg16, pretrained_path)


        # self.vgg16.load_state_dict(torch.load(pretrained_path),strict=True) 
        print(self.vgg16)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.model_feature = nn.Sequential(*list(model.children())[:-1])
        # self.last_linear = nn.Linear(fc_features, class_number) 
            

        
    def forward(self, img):        
        out = self.vgg16(img)
        # out = self.last_linear(out)
        #out = self.avgpool(out)

        return out





class SSD(nn.Module):
    def __init__(self, num_class, pretrained_path=None):
        super(SSD, self).__init__()

        self.num_class = num_class
        

        backbone = VGG16()
        if pretrained_path:
            backbone.load_state_dict(torch.load(pretrained_path),strict=True) 

        # print(backbone)

        self.num_prior_box = [4,6,6,6,4,4]
        

        self.feature_channel = [512, 1024, 512, 256, 256, 256]

        self.features_layer1 = nn.Sequential(*list(backbone.features.children())[:23])

        self.features_layer2 = nn.Sequential(
                                    *list(backbone.features.children())[23:-1],
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, kernel_size=1),
                                    nn.ReLU(inplace=True)
                                    )

        #额外新增的4个特征层
        self.features_layer3 = nn.Sequential(
                                    nn.Conv2d(1024, 256, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(inplace=True),
                                    )

        self.features_layer4 = nn.Sequential(
                                    nn.Conv2d(512, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                    nn.ReLU(inplace=True),
                                    )

        self.features_layer5 = nn.Sequential(
                                    nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True),
                                    )

        self.features_layer6 = nn.Sequential(
                                    nn.Conv2d(256, 128, kernel_size=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True),
                                    )
        self.L2Norm = L2Norm(512, 20)

        self.classifier_layer1, self.location_layer1 = self.buildHeader(self.num_prior_box[0], self.feature_channel[0])
        self.classifier_layer2, self.location_layer2 = self.buildHeader(self.num_prior_box[1], self.feature_channel[1])
        self.classifier_layer3, self.location_layer3 = self.buildHeader(self.num_prior_box[2], self.feature_channel[2])
        self.classifier_layer4, self.location_layer4 = self.buildHeader(self.num_prior_box[3], self.feature_channel[3])
        self.classifier_layer5, self.location_layer5 = self.buildHeader(self.num_prior_box[4], self.feature_channel[4])
        self.classifier_layer6, self.location_layer6 = self.buildHeader(self.num_prior_box[5], self.feature_channel[5])

        # self.detection_output_layer  =

        #print(self.features_layer1)
        # b
        # self.features1 = backbone.features1
        # self.features2 = backbone.features2
        # self.features3 = backbone.features3
        # self.features4 = backbone.features4
        # self.features5 = backbone.features5
        # self.features = backbone.features
        # self.avgpool = backbone.avgpool
        # self.classifier = backbone.classifier

    def buildHeader(self, num_prior_box, feature_channel):
        classifier_layer = nn.Sequential(
                                    nn.Conv2d(feature_channel, num_prior_box*(self.num_class+1), kernel_size=3, padding=1),
                                    # nn.ReLU(inplace=True),
                                    )
        location_layer = nn.Sequential(
                                    nn.Conv2d(feature_channel, num_prior_box*4, kernel_size=3, padding=1),
                                    # nn.ReLU(inplace=True),
                                    )
        return classifier_layer, location_layer
        
    def forward(self, img):        
        features1 = self.features_layer1(img) # [n, 512, 38, 38]
        #https://blog.csdn.net/loovelj/article/details/106556851
        features2 = self.features_layer2(features1) # [n, 1024, 19, 19]
        features3 = self.features_layer3(features2) # [n, 512, 10, 10]
        features4 = self.features_layer4(features3) # [n, 256, 5, 5]
        features5 = self.features_layer5(features4) # [n, 256, 3, 3]
        features6 = self.features_layer6(features5) # [n, 256, 1, 1]

        features1 = self.L2Norm(features1)
        classifier1 = self.classifier_layer1(features1)
        classifier1 = classifier1.permute(0, 2, 3, 1).contiguous()
        classifier1 = classifier1.view(classifier1.size(0),-1,classifier1.size(3)//4)
        location1 = self.location_layer1(features1)
        location1 = location1.permute(0, 2, 3, 1).contiguous()
        location1 = location1.view(location1.size(0),-1,location1.size(3)//4)

        classifier2 = self.classifier_layer2(features2)
        classifier2 = classifier2.permute(0, 2, 3, 1).contiguous()
        classifier2 = classifier2.view(classifier2.size(0),-1,classifier2.size(3)//6)
        location2 = self.location_layer2(features2)
        location2 = location2.permute(0, 2, 3, 1).contiguous()
        location2 = location2.view(location2.size(0),-1,location2.size(3)//6)


        classifier3 = self.classifier_layer3(features3)
        classifier3 = classifier3.permute(0, 2, 3, 1).contiguous()
        classifier3 = classifier3.view(classifier3.size(0),-1,classifier3.size(3)//6)
        location3 = self.location_layer3(features3)
        location3 = location3.permute(0, 2, 3, 1).contiguous()
        location3 = location3.view(location3.size(0),-1,location3.size(3)//6)

        classifier4 = self.classifier_layer4(features4)
        classifier4 = classifier4.permute(0, 2, 3, 1).contiguous()
        classifier4 = classifier4.view(classifier4.size(0),-1,classifier4.size(3)//6)
        location4 = self.location_layer4(features4)
        location4 = location4.permute(0, 2, 3, 1).contiguous()
        location4 = location4.view(location4.size(0),-1,location4.size(3)//6)

        classifier5 = self.classifier_layer5(features5)
        classifier5 = classifier5.permute(0, 2, 3, 1).contiguous()
        classifier5 = classifier5.view(classifier5.size(0),-1,classifier5.size(3)//4)
        location5 = self.location_layer5(features5)
        location5 = location5.permute(0, 2, 3, 1).contiguous()
        location5 = location5.view(location5.size(0),-1,location5.size(3)//4)

        classifier6 = self.classifier_layer6(features6)
        classifier6 = classifier6.permute(0, 2, 3, 1).contiguous()
        classifier6 = classifier6.view(classifier6.size(0),-1,classifier6.size(3)//4)
        location6 = self.location_layer6(features6)
        location6 = location6.permute(0, 2, 3, 1).contiguous()
        location6 = location6.view(location6.size(0),-1,location6.size(3)//4)

        # print(classifier1.shape, location1.shape)
        # print(classifier2.shape, location2.shape)
        # print(classifier3.shape, location3.shape)
        # print(classifier4.shape, location4.shape)
        # print(classifier5.shape, location5.shape)
        # print(classifier6.shape, location6.shape)
        """
        torch.Size([1, 84, 38, 38]) torch.Size([1, 16, 38, 38])
        torch.Size([1, 126, 19, 19]) torch.Size([1, 24, 19, 19])
        torch.Size([1, 126, 10, 10]) torch.Size([1, 24, 10, 10])
        torch.Size([1, 126, 5, 5]) torch.Size([1, 24, 5, 5])
        torch.Size([1, 84, 3, 3]) torch.Size([1, 16, 3, 3])
        torch.Size([1, 84, 1, 1]) torch.Size([1, 16, 1, 1])
        """

        confidence = torch.cat([classifier1,classifier2,classifier3,classifier4,classifier5,classifier6], 1)
        location = torch.cat([location1,location2,location3,location4,location5,location6], 1)
        #print(classifier.shape, location.shape)#torch.Size([1, 8732, 21]) torch.Size([1, 8732, 4])
        # b
        return confidence, location




def funcL2Norm(inputs):
    eps = 1e-10
    norm = inputs.pow(2).sum(dim=1, keepdim=True).sqrt()+eps
    out = torch.div(inputs,norm)
    return out


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out



if __name__ == "__main__":
    device = torch.device("cuda")
    classes = 1000
    model = SSD(classes).to(device)
    print(model)

    


    