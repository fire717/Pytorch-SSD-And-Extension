# Pytorch-SSD-And-Extension

### Env
* python3
* torch 1.7.0

### 验证

对比项目 [
ssd.pytorch（下表简称SP）](https://github.com/amdegroot/ssd.pytorch)，本项目简称PSAE

测试集：VOC2007 Test

* SP（VOC2007tranval+2012 batch_size24 带数据增强）：mAP 0.7728 （iter115000）
* SP（VOC2007tranval+2012 batch_size24 不带数据增强）：mAP 0.6156 （iter115000）
* SP（只用VOC2007tranval batch_size24 带数据增强）：mAP 0.7036 （iter115000）
* SP（只用VOC2007tranval batch_size24 不带数据增强）：mAP 0.4397（iter115000）（0.5243 iter5000）数据少过拟合了 
* SP（只用VOC2007tranval batch_size24 不带数据增强 不加载预训练模型）：
* PSAE（只用VOC2007tranval batch_size24 不带数据增强）：

### To Do

- [x] base (loss func)
- [ ] mAP评测
- [ ] 对比数据预处理resize、crop、pad的效果
- [ ] add relu or not
- [ ] 参数初始化
- [ ] other loss function
- [ ] other nms
- [ ] other structure（RFB、轻量级模型）
- [ ] 单目标检测效果对比
- [ ] ...


### Ref

* [SSD: Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)
* [SSD网络结构的理解](https://zhuanlan.zhihu.com/p/148539724)
* [SSD300默认框尺寸计算](https://blog.csdn.net/qq_37116150/article/details/105794992)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
