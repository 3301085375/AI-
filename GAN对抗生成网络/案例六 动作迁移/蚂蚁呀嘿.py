############### 图像大小要能被16整除##############################

# 蚂蚁呀嘿实现步骤三步走：
# 下载PaddleGAN
# 运行First Order Motion命令
# 配上音乐🎵

# 蚂蚁呀嘿技术流程
# 整体流程分为三步：

# 将照片中的多人脸使用人脸检测模型S3FD框出并抠出
# 对抠出的人脸用First Order Motion进行脸部表情迁移
# 将迁移好的人脸放回对应的原位置

# 从github上克隆PaddleGAN代码
#!git clone https://github.com/PaddlePaddle/PaddleGAN
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
%cd PaddleGAN
!git checkout develop

# 安装所需安装包
!pip install -r requirements.txt
!pip install imageio-ffmpeg
%cd applications/

###############################################################################################################################################################
# 大家可以上传自己准备的视频和图片，并在如下命令中的source_image参数和driving_video参数分别换成自己的图片和视频路径，然后运行如下命令，就可以完成动作表情迁移，程序运行成功后，会在ouput文件夹生成名为result.mp4的视频文件，该文件即为动作迁移后的视频。

# 同时，根据自己上传的照片中人脸的间距，

# 本项目中提供了原始图片和驱动视频供展示使用。具体的各参数使用说明如下

# driving_video: 驱动视频，视频中人物的表情动作作为待迁移的对象
# source_image: 原始图片，视频中人物的表情动作将迁移到该原始图片中的人物上
# relative: 指示程序中使用视频和图片中人物关键点的相对坐标还是绝对坐标，建议使用相对坐标，若使用绝对坐标，会导致迁移后人物扭曲变形
# adapt_scale: 根据关键点凸包自适应运动尺度
# ratio：将框出来的人脸贴回原图时的区域占宽高的比例，默认为0.4，范围为【0.4，0.5】
#############################################################################################################################
!export PYTHONPATH=$PYTHONPATH:/home/aistudio/PaddleGAN && python -u tools/first-order-demo.py  --driving_video ~/work/2.MP4  --source_image ~/work/yiyi.jpg --ratio 0.4 --relative --adapt_scale 
  
# 最后一步：使用moviepy为生成的视频加上音乐
# add audio
!pip install moviepy

#为生成的视频加上音乐
from moviepy.editor import *

videoclip_1 = VideoFileClip("/home/aistudio/work/2.MP4")
videoclip_2 = VideoFileClip("./output/result.mp4")

audio_1 = videoclip_1.audio

videoclip_3 = videoclip_2.set_audio(audio_1)
videoclip_3.write_videofile("./output/finalout.mp4", audio_codec="aac")
