# <iframe src="//www.bilibili.com/video/BV1254y1L7Kz/" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

%cd /home/aistudio/work/

# 从github上克隆PaddleGAN代码（如下载速度过慢，可用gitee源）
!git clone https://gitee.com/PaddlePaddle/PaddleGAN
#!git clone https://github.com/PaddlePaddle/PaddleGAN

%cd /home/aistudio/work/PaddleGAN

!pip install -r requirements.txt
%cd applications/

!export PYTHONPATH=$PYTHONPATH:/home/aistudio/work/PaddleGAN && python tools/wav2lip.py --face /home/aistudio/work/光头强.PNG --audio /home/aistudio/work/音频1.mp3 --outfile /home/aistudio/work/光头强唱歌.mp4
  
