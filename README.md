# Object-Detection-Papers-and-Reading-Notes
Object Detection Papers and Reading Notes

![1](https://github.com/ice609/Object-Detection-Papers-and-Reading-Notes/blob/master/deep_learning_object_detection_history.PNG)


深度学习在图像领域的应用
随着深度学习近几年的火热发展，在计算机视觉，图像理解方向上，应用越来越广泛。我们总结了在视觉领域的一些方向上基于深度学习的优秀算法。包括物体检测、物体识别、人脸世界、分割、跟踪、边缘检测、图像复原（去雨、去雾）、图像编辑等。


检测

1. 单一物体检测

MTCNN: https://github.com/kpzhang93/MTCNNfacedetection_alignment

Cascade-CNN: https://github.com/anson0910/CNNfacedetection

2. 通用物体检测

Faster-RCNN: https://github.com/rbgirshick/py-faster-rcnn

YOLO: https://github.com/pjreddie/darknet

SSD: https://github.com/FreeApe/VGG-or-MobileNet-SSD

RetinaNet: https://github.com/fizyr/keras-retinanet

分类

VGG: https://github.com/ry/tensorflow-vgg16

GoogLenet: https://github.com/n3011/Inceptionv3GoogLeNet

Resnet: https://github.com/ry/tensorflow-resnet

Mobilenet: https://github.com/shicai/MobileNet-Caffe

Shufflenet: https://github.com/anlongstory/ShuffleNet_V2-caffe

MNasNet: https://github.com/zeusees/Mnasnet-Pretrained-Model

识别

1. 人脸识别

Deepface: https://github.com/RiweiChen/DeepFace

Normface: https://github.com/happynear/NormFace

Insightface: https://github.com/deepinsight/insightface

2. 文字识别

DeepOCR: https://github.com/JinpengLI/deep_ocr

CTPN: https://github.com/tianzhi0549/CTPN (文字定位)

DenseNet + CTC: https://github.com/YCG09/chinese_ocr

YOLOv3 + CRNN : https://github.com/chineseocr/chineseocr

跟踪

1.
2.

分割

Unet: https://github.com/zhixuhao/unet

mask-rcnn: https://github.com/matterport/Mask_RCNN

边缘检测

HED: https://github.com/s9xie/hed

RCF: https://github.com/yun-liu/rcf

图像复原

1. 去雨

DDN: https://github.com/XMU-smartdsp/Removing_Rain

CGAN: https://github.com/hezhangsprinter/ID-CGAN

DID-MDN: https://github.com/hezhangsprinter/DID-MDN

DeRaindrop: https://github.com/rui1996/DeRaindrop

2. 去雾

MSCNN: https://github.com/dishank-b/MSCNN-Dehazing-Tensorflow

DehazeNet: https://github.com/caibolun/DehazeNet

3. 超分辨率

SRCNN: https://github.com/tegg89/SRCNN-Tensorflow

EDSR: https://github.com/thstkdgus35/EDSR-PyTorch (https://blog.csdn.net/xjp_xujiping/article/details/81986020)

4.图像单反化

DPED: https://github.com/aiff22/DPED

总结

目前深度学习技术在计算机视觉算法、图像处理算法运用越来越广泛，这里把我们在工程中常用的一些网络加以整理总结，方便后面的使用者学习。在很多细分领域，深度学习同样发挥了巨大作用，例如医学领域，自然语言处理等，由于这些领域专业性更强，通常是多学科的结合，我们应用不多，没办法为大家提供详细的研究材料，大家见谅。



算法总结：

RCNN 

　　1. 在图像中确定约1000-2000个候选框 (使用选择性搜索) 
  
　　2. 每个候选框内图像块缩放至相同大小，并输入到CNN内进行特征提取 
  
　　3. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类 
  
　　4. 对于属于某一特征的候选框，用回归器进一步调整其位置 
  
Fast RCNN 

　　1. 在图像中确定约1000-2000个候选框 (使用选择性搜索) 
  
　　2. 对整张图片输进CNN，得到feature map 
  
　　3. 找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入SPP layer和之后的层 
  
　　4. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类 
  
　　5. 对于属于某一特征的候选框，用回归器进一步调整其位置 
  
Faster RCNN 

　　1. 对整张图片输进CNN，得到feature map 
  
　　2. 卷积特征输入到RPN，得到候选框的特征信息 
  
　　3. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类 
  
　　4. 对于属于某一特征的候选框，用回归器进一步调整其位置


Fast RCNN  ROI Pooling的输出

输出是batch个vector，其中batch的值等于RoI的个数，vector的大小为channel * w * h；RoI Pooling的过程就是将一个个大小不同的box矩形框，都映射成大小固定（w * h）的矩形框


R-FCN结构的优点

　　R-FCN要解决的根本问题是Faster R-CNN检测速度慢的问题，速度慢是因为ROI层后的结构对不同的proposal是不共享的，试想下如果有300个proposal，ROI后的全连接网络就要计算300次，这个耗时就太吓人了。所以作者把ROI后的结构往前挪来提升速度，但光是挪动下还不行，ROI在conv5后会引起上节提到的平移可变性问题，必须通过其他方法加强结构的可变性，所以作者就想出了通过添加Position-sensitive score map来达到这个目的。 


One-stage缺点

   One-stage受制于万恶的 “类别不平衡” 。
   
   1.什么是“类别不平衡”呢？

   详细来说，检测算法在早期会生成一大波的bbox。而一幅常规的图片中，顶多就那么几个object。这意味着，绝大多数的bbox属于background。

   2.“类别不平衡”又如何会导致检测精度低呢？

   因为bbox数量爆炸。 
   正是因为bbox中属于background的bbox太多了，所以如果分类器无脑地把所有bbox统一归类为background，accuracy也可以刷得很高。于是乎，分类器的训练就失败了。分类器训练失败，检测精度自然就低了。

   3.那为什么two-stage系就可以避免这个问题呢？

   因为two-stage系有RPN罩着。 
   第一个stage的RPN会对anchor进行简单的二分类（只是简单地区分是前景还是背景，并不区别究竟属于哪个细类）。经过该轮初筛，属于background的bbox被大幅砍削。虽然其数量依然远大于前景类bbox，但是至少数量差距已经不像最初生成的anchor那样夸张了。就等于是 从 “类别 极 不平衡” 变成了 “类别 较 不平衡” 。 
不过，其实two-stage系的detector也不能完全避免这个问题，只能说是在很大程度上减轻了“类别不平衡”对检测精度所造成的影响。 
接着到了第二个stage时，分类器登场，在初筛过后的bbox上进行难度小得多的第二波分类(这次是细分类)。这样一来，分类器得到了较好的训练，最终的检测精度自然就高啦。但是经过这么两个stage一倒腾，操作复杂，检测速度就被严重拖慢了。

   4.那为什么one-stage系无法避免该问题呢？

   因为one stage系的detector直接在首波生成的“类别极不平衡”的bbox中就进行难度极大的细分类，意图直接输出bbox和标签（分类结果）。而原有交叉熵损失(CE)作为分类任务的损失函数，无法抗衡“类别极不平衡”，容易导致分类器训练失败。因此，one-stage detector虽然保住了检测速度，却丧失了检测精度。

FCN

   FCN提出可以把后面几个全连接都换成卷积，这样就可以获得一张2维的feature map，后接softmax获得每个像素点的分类信息，从而解决了分割问题。
   
   FCN对图像进行像素级的分类，从而解决了语义级别的图像分割（semantic segmentation）问题。与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类（全联接层＋softmax输出）不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。 下图是语义分割所采用的全卷积网络(FCN)的结构示意图：
   
   ![1](https://github.com/ice609/Object-Detection-Papers-and-Reading-Notes/blob/master/FCN.PNG)

新增综述

深度学习目标检测方法综述-CSDN    https://blog.csdn.net/zong596568821xp/article/details/80091784  (recommend)


YOLO系列	

		
1	目标检测-Blog	http://www.cnblogs.com/cloud-ken/p/9470992.html

2	综述|基于深度学习的目标检测(一)-小将	https://zhuanlan.zhihu.com/p/34325398

3	YOLO原理与实现-小将	https://zhuanlan.zhihu.com/p/32525231

4	YOLOv2原理与实现(附YOLOv3)-小将	https://zhuanlan.zhihu.com/p/35325884

5	YOLOv3论文笔记-CSDN	https://blog.csdn.net/cherry_yu08/article/details/81102049

6	目标检测|SSD原理与实现-小将	https://zhuanlan.zhihu.com/p/33544892

7	目标检测算法之YOLO-CSDN	https://blog.csdn.net/SIGAI_CSDN/article/details/80718050

8	YOLOv3-量子位	https://mp.weixin.qq.com/s/cq7g1-4oFTftLbmKcpi_aQ
        
   代码   https://mp.weixin.qq.com/s/6VfxXnm_BERubvCTWLfHBg

9	YOLO-官网	https://pjreddie.com/darknet/yolo/

10	YOLO v2-B站Mark Jay	https://www.bilibili.com/video/av20946619?from=search&seid=144843201247986672

11	YOLO 9000-B站YOLO v2 author	https://www.bilibili.com/video/av16183738?from=search&seid=11347358533356221986

12	YOLO及YOLOv2-B站Monenta Paper解读	https://www.bilibili.com/video/av31974438/

13	基于深度学习的目标检测技术-B站	https://www.bilibili.com/video/av19359393/?spm_id_from=333.788.videocard.4

14	Fast AI（Object detection）-B站	https://www.bilibili.com/video/av23172110?from=search&seid=11969804202491535833

15	Fast AI（Object detection）-B站	https://www.bilibili.com/video/av10156946?from=search&seid=15700525747364222189

16	目标检测和边界框-李沐动手深度学习	http://zh.gluon.ai/chapter_computer-vision/bounding-box.html

17	YOLO V2 Keras- Github	https://github.com/yhcc/yolo2

18	深度学习目标检测算法综述-AI研习社	https://mp.weixin.qq.com/s/l8EWM_ItwDrT5N8tj-STWA

19      CenterNet     https://github.com/Duankaiwen/CenterNet




R-CNN系列	

		
1	从R-CNN到Mask R-CNN-知乎	https://zhuanlan.zhihu.com/p/30967656

2	R-CNN论文详解-CSDN	https://blog.csdn.net/wopawn/article/details/52133338

3	SPP-Net论文详解-CSDN	https://blog.csdn.net/v1_vivian/article/details/73275259

4	SPP-Net论文翻译-Blog	http://www.dengfanxin.cn/?p=403

5	ROI Pooling原理及实现-CSDN	https://blog.csdn.net/u011436429/article/details/80279536

6	ROI Pooling层解析-CSDN	https://blog.csdn.net/lanran2/article/details/60143861

7	Faster R-CNN 中 RPN 原理-CSDN	https://blog.csdn.net/zziahgf/article/details/79895804

8	 RPN (区域候选网络)-CSDN	https://blog.csdn.net/JNingWei/article/details/78847696

9	Faster-RCNN算法精读-CSDN	https://blog.csdn.net/hunterlew/article/details/71075925

10      Faster-rcnn详解-CSDN     https://blog.csdn.net/WZZ18191171661/article/details/79439212

11      Faster-rcnn详解-CSDN     https://blog.csdn.net/zziahgf/article/details/79311275  (recommend)

12	ROIPooling和ROIAlign对比-Blog	https://baijiahao.baidu.com/s?id=1616632836625777924&wfr=spider&for=pc

13	Mask R-CNN详解-CSDN	https://blog.csdn.net/WZZ18191171661/article/details/79453780

R-FCN

1       综述（Faster R-CNN、R-FCN和SSD）-机器之心        https://www.jiqizhixin.com/articles/2017-09-18-7

2       R-FCN讲解-CSDN      https://blog.csdn.net/LeeWanzhi/article/details/79770376

FPN

作者提出的FPN（Feature Pyramid Network）算法同时利用低层特征高分辨率和高层特征的高语义信息，通过融合这些不同层的特征达到预测的效果。并且预测是在每个融合后的特征层上单独进行的，这和常规的特征融合方式不同。

FPN是对用卷积神经网络进行目标检测方法的一种改进，通过提取多尺度的特征信息进行融合，进而提高目标检测的精度，特别是在小物体检测上的精度。

文章的思想主要是利用特征金字塔对不同层次的特征进行尺度变化后，再进行信息融合，从而可以提取到比较低层的信息，也就是相对顶层特征来说更加详细的信息。顶层特征在不断地卷积池化过程中可能忽略了小物体的一些信息，特征金字塔通过不同层次的特征融合，使得小物体的信息也能够比较完整地反映出来。这个方法可以广泛地应用在针对小目标物体的检测上。


1       FPN（feature pyramid networks）算法讲解-CSDN     https://blog.csdn.net/u014380165/article/details/72890275/

2       FPN特征金字塔网络--论文解读-CSDN      https://blog.csdn.net/weixin_40683960/article/details/79055537


RefineDet

1      RefineDet算法笔记-CSDN    https://blog.csdn.net/u014380165/article/details/79502308

2      论文笔记——RefineDet-CSDN    https://blog.csdn.net/nwu_nbl/article/details/81110286


RetinaNet

1      论文阅读: RetinaNet-CSDN     https://blog.csdn.net/JNingWei/article/details/80038594

深度学习模型压缩方法

1     深度学习模型压缩方法综述-CSDN   https://blog.csdn.net/wspba/article/details/75671573

2     当前深度神经网络模型压缩和加速方法速览-CSDN   https://blog.csdn.net/Touch_Dream/article/details/78441332


视频

Faster R-CNN

1      【 Faster R-CNN 】Paper Review  https://www.bilibili.com/video/av15949356?from=search&seid=13647145201383642188

Mask R-CNN

0      作者详细讲解YOLO-B站     https://www.bilibili.com/video/av11200546/?spm_id_from=trigger_reload

1      YOLO 目标检测 (TensorFlow tutorial)-B站    https://www.bilibili.com/video/av37448440

2      深度学习顶级论文算法详解-B站          https://www.bilibili.com/video/av30271782/?p=5

3      将门分享-任少卿-From Faster R-CNN to Mask R-CNN-B站    https://www.bilibili.com/video/av19507321

4      将门 | 旷视俞刚-Beyond RetinaNet & Mask R-CNN-B站   https://www.bilibili.com/video/av29340771  (recommend)

5      Momenta Paper Reading Mask R-CNN解读-B站     https://www.bilibili.com/video/av31977792

6      Fast AI Object Detection-B站      https://www.bilibili.com/video/av24120121

7      何凯明 Mask R-CNN ICCV 2017-B站     https://www.bilibili.com/video/av21410129?from=search&seid=2835437329822299200

8      Paper Review Mask RCNN-B站          https://www.bilibili.com/video/av15949583?from=search&seid=2835437329822299200

9      Mask RCNN with Keras and Tensorflow-B站      https://www.bilibili.com/video/av20821618?from=search&seid=2835437329822299200

Object Detection资源汇总链接

https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html


