

# 用主流的GAN制作编码器，并编辑高清人脸图片


  <img src="./images/cxx1.gif" width = "128" height = "128" alt="cxx1"  />  <img src="./images/cxx2.gif" width = "128" height = "128" alt="cxx2"  />  <img src="./images/msk.gif" width = "128" height = "128" alt="msk" />   <img src="./images/dy.gif" width = "128" height = "128" alt="dy" />  <img src="./images/zy.gif" width = "128" height = "128" alt="zy" /> 

看到这几幅人脸动画图了吧，这些都是通过一张图片生成。本篇博客介绍一种简练的方法，即用Stylegan(V1)编辑图片并改变属性 (如让人脸微笑，带眼镜，转方向等).

## 1.加载StyleGAN (v1)

首先需要下载预训练模型，保重StyleGAN 1024x1024的人脸图像能生成 (即复现StyleGAN).

这里需要注意的是，预训练模型保存为三个部分:

a.Gm, 将z [n, 512]映射为w [n, 18, 512]. 其中n是batch_size

b.Gs, 输入w 输出对应图片, 注意这个w是每层输入一个[n,512]. 有18层, 即 [n,18,512]

c.avg_tensor, 这个是一个训练好的常向量，用于模型的首次输入,[n, 512,4 , 4]. (而上面的w是从各个层单独输入)

这里提供StyleGANv1预训练模型 [Model](https://pan.baidu.com/s/1_JewahCd_UK5wIMCQzFAPA 
) , 提取码: kwsk

## 2.将真实图片编码

通过一个我们提供的编码器将一张1024X1024的人脸图片编码到潜变量W  (1,18,512) ,  也可以同时处理多张人脸 (1->n)，这个根据自己显卡内存大小决定。另外需要注意最好是五官对齐的人脸.  并把编好的W保存到文件夹(默认：./latentvectors/faces/). 里面已经有多张人脸了，可以用于测试上面的StyleGAN. 

可以运行以下文件：

> python embedding_img.py

a.这个文件加载Encoder预训练模型
> 默认路径为./checkpoint/E/E_blur(case2)_styleganv1_FFHQ_state_dict.pth

b.并编码真实图像
>默认路径为 ./checkpoint/realimg_file/)编码到w (默认路径为 ./result/models/

c.这里提供Encoder预训练模型 [Model](https://pan.baidu.com/s/1F9Tv5ph9Rejp5JTQK2HSYQ 
) , 提取码: swtl

## 3. 编辑表情

用'./latentvectors/directions/'中的属性向量乘以一个系数(5或10的倍数)，加上人脸的向量，探索并保存人脸变化，参见:

> embedded_img_processing.py

如果想要探索其他人脸属性，也很多方法，最简单有监督方法可以参考: https://github.com/Puzer/stylegan-encoder (里面也附带人脸对齐)






