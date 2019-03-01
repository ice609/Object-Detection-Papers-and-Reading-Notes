# Object-Detection-Papers-and-Reading-Notes
Object Detection Papers and Reading Notes

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
 

R-FCN结构的优点

　　R-FCN要解决的根本问题是Faster R-CNN检测速度慢的问题，速度慢是因为ROI层后的结构对不同的proposal是不共享的，试想下如果有300个proposal，ROI后的全连接网络就要计算300次，这个耗时就太吓人了。所以作者把ROI后的结构往前挪来提升速度，但光是挪动下还不行，ROI在conv5后会引起上节提到的平移可变性问题，必须通过其他方法加强结构的可变性，所以作者就想出了通过添加Position-sensitive score map来达到这个目的。 




YOLO系列	

		
1	目标检测-Blog	http://www.cnblogs.com/cloud-ken/p/9470992.html

2	综述|基于深度学习的目标检测(一)-小将	https://zhuanlan.zhihu.com/p/34325398

3	YOLO原理与实现-小将	https://zhuanlan.zhihu.com/p/32525231

4	YOLOv2原理与实现(附YOLOv3)-小将	https://zhuanlan.zhihu.com/p/35325884

5	YOLOv3论文笔记-CSDN	https://blog.csdn.net/cherry_yu08/article/details/81102049

6	目标检测|SSD原理与实现-小将	https://zhuanlan.zhihu.com/p/33544892

7	目标检测算法之YOLO-CSDN	https://blog.csdn.net/SIGAI_CSDN/article/details/80718050

8	YOLOv3-量子位	https://mp.weixin.qq.com/s/cq7g1-4oFTftLbmKcpi_aQ

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

1       综述-机器之心        https://www.jiqizhixin.com/articles/2017-09-18-7

2       R-FCN讲解-CSDN      https://blog.csdn.net/LeeWanzhi/article/details/79770376
