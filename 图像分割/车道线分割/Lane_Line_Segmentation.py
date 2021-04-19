##################使用PaddleSeg进行车道线分割###########################

#  从 7000轮的基础上训练，过程可能要十个多小时，

#                       终端执行  
# 解压数据集
# 将数据集提前放好位置 
!unzip data/data68698/智能车数据集.zip

# 从官网克隆 PaddleSeg 工具包
!git clone https://gitee.com/paddlepaddle/PaddleSeg.git
  
# 准备数据集
# PaddleSeg目前支持CityScapes、ADE20K、Pascal VOC等数据集的加载，在加载数据集时，如若本地不存在对应数据，则会自动触发下载(除Cityscapes数据集)。 这里可以直接使用比赛提供的脚本
%run make_list.py
# !python make_list.py

%set_env CUDA_VISIBLE_DEVICES=0

%cd PaddleSeg

# 训练
# 可以选择任意config文件重新训练
# !python train.py \
#        --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k_lovasz_softmax.yml \
#        --do_eval \
#        --use_vdl \
#        --save_interval 750 \
#        --save_dir output

# 也可以在项目提供的7000轮训练结果上继续训练
!python train.py \
       --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k_lovasz_softmax.yml \
       --resume_model output/iter_7000 \
       --do_eval \
       --use_vdl \
       --save_interval 1000 \
       --save_dir output

# 评估
!python val.py \
       --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k_lovasz_softmax.yml \
       --model_path output/iter_7000/model.pdparams

# 预测
!python predict.py \
       --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k_lovasz_softmax.yml \
       --model_path output/iter_7000/model.pdparams \
       --image_path ../infer/4346.png \
       --save_dir output/result

# 导出静态图模型
!python export.py \
       --config configs/ocrnet/ocrnet_hrnetw18_cityscapes_1024x512_160k_lovasz_softmax.yml \
       --model_path output/iter_7000/model.pdparams

# 可视化结果
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('../infer/4346.png',0)
label = cv2.imread('../PaddleSeg/output/result/pseudo_color_prediction/4346.png',0)
plt.figure(figsize=(10,5))
plt.subplot(121)
# 打印4346.png
plt.imshow(image)
plt.subplot(122)
# 伪彩色图像(pseudo_color)
plt.imshow(label)
plt.show()
