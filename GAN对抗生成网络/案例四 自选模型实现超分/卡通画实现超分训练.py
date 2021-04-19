# # ############## 对应的 在 3301085375/PaddleGAN/education/文件夹下也有更详细的作业过程记录。

#   通过自己在配置文件当中修改相关参数设置，实现用自己选用的模型训练，并得到模型对卡通图像进行超分辨率转化。

# 安装PaddleGAN
# PaddleGAN的安装目前支持Clone GitHub和Gitee两种方式：
# 安装ppgan
# 当前目录在: /home/aistudio/, 这个目录也是左边文件和文件夹所在的目录
# 克隆最新的PaddleGAN仓库到当前目录
# !git clone https://github.com/PaddlePaddle/PaddleGAN.git
# 如果从github下载慢可以从gitee clone：
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
# 安装Paddle GAN
%cd PaddleGAN/
!pip install -v -e .


# 数据准备
# 我们为大家准备了处理好的超分数据集卡通画超分数据集
# 回到/home/aistudio/下
%cd /home/aistudio
# 解压数据
!unzip -q data/data80790/animeSR.zip -d data/
# 将解压后的数据链接到` /home/aistudio/PaddleGAN/data `目录下
!mv data/animeSR PaddleGAN/data/

# 数据可视化
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 训练数据统计
train_names = os.listdir('PaddleGAN/data/animeSR/train')
print(f'训练集数据量: {len(train_names)}')

# 测试数据统计
test_names = os.listdir('PaddleGAN/data/animeSR/test')
print(f'测试集数据量: {len(test_names)}')

# 训练数据可视化
img = cv2.imread('PaddleGAN/data/animeSR/train/Anime_1.jpg')
img = img[:,:,::-1]
plt.figure()
plt.imshow(img)
plt.show()

# 选择超分模型
# PaddleGAN中提供的超分模型包括RealSR, ESRGAN, LESRCNN, DRN等，详情可见超分模型。

# 接下来以ESRGAN为例进行演示。
################################################################################################
# 修改配置文件
# 所有模型的配置文件均在/home/aistudio/PaddleGAN/configs目录下。

# 找到你需要的模型的配置文件，修改模型参数，一般修改迭代次数，num_workers，batch_size以及数据集路径。有能力的同学也可以尝试修改其他参数，或者基于现有模型进行二次开发，模型代码在/home/aistudio/PaddleGAN/ppgan/models目录下。

# 以ESRGAN为例，这里将将配置文件esrgan_psnr_x4_div2k.yaml中的

# 参数total_iters改为50000      我在本案例中选用30000

# 参数dataset：train：num_workers改为12

# 参数dataset：train：batch_size改为48    我在本案例中选用 6，太大了会导致内存溢出，系统奔溃。

# 参数dataset：train：gt_folder改为data/animeSR/train

# 参数dataset：train：lq_folder改为data/animeSR/train_X4

# 参数dataset：test：gt_folder改为data/animeSR/test

# 参数dataset：test：lq_folder改为data/animeSR/test_X4

# # 训练模型
# # 以ESRGAN为例，运行以下代码训练ESRGAN模型。

# # 如果希望使用其他模型训练，可以修改配置文件名字。
# %cd /home/aistudio/PaddleGAN/
# !python -u tools/main.py --config-file configs/esrgan_psnr_x4_div2k.yaml

#################################### 自己的代码 ###########################
%cd /home/aistudio/PaddleGAN/
!python -u tools/main.py --config-file configs/lesrcnn_psnr_x4_div2k.yaml

# 测试模型
# 以ESRGAN为例，模型训练好后，运行以下代码测试ESRGAN模型。

# 其中/home/aistudio/pretrained_model/ESRGAN_PSNR_50000_weight.pdparams是刚才ESRGAN训练的模型参数，同学们需要换成自己的模型参数。

# # 如果希望使用其他模型测试，可以修改配置文件名字。
# %cd /home/aistudio/PaddleGAN/
# !python tools/main.py --config-file configs/esrgan_psnr_x4_div2k.yaml --evaluate-only --load /home/aistudio/pretrained_model/ESRGAN_PSNR_50000_weight.pdparams

#################################### 自己的代码 ###########################
%cd /home/aistudio/PaddleGAN/
!python tools/main.py --config-file configs/lesrcnn_psnr_x4_div2k.yaml --evaluate-only --load /home/aistudio/pretrained_model/ESRGAN_PSNR_50000_weight.pdparams
