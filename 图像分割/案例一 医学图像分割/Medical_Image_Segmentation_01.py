#  数据用是 LITS 肝脏数据集28个病例，使用PaddleSeg开发套件进行分割肝脏和肿瘤

# 训练2500次要8分钟左右。  mIoU达到0.7855.

###############################################可以在终端永久安装##################################
#  安装 nii处理工具  SimpleITK 和分割工具paddleSeg
!pip install SimpleITK
!pip install paddleseg

%cd /home/aistudio/
# 首次要运行
!unzip  -o /home/aistudio/data/data79322/traindata.zip -d work/  # -o 是覆盖解压

##################### 下面代码可以在终端执行，也可以在IDE执行
#导入常用库
import SimpleITK as sitk
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import cv2

# 2.处理数据，因为数据集里面有些是全腹部的，但是要分割肝脏，所以把不存在肝脏标签的多余‘层’去掉。
raw_dataset_path = 'work/traindata'
# 预处理后的数据集的输出路径
fixed_dataset_path = 'work/new_traindata'
if not os.path.exists(fixed_dataset_path):
    os.mkdir(fixed_dataset_path)

 # 非首次执行该cell时，这里会报错，所以首次执行结束后可以将这部分代码注释掉
if os.path.exists(fixed_dataset_path):    # 创建保存目录
    os.makedirs(os.path.join(fixed_dataset_path,'data'))
    os.makedirs(os.path.join(fixed_dataset_path,'label'))

upper = 200   #上阈值
lower = -200  #下阈值
for ct_file in os.listdir(os.path.join(raw_dataset_path ,'data')):
    #读取origin
    ct = sitk.ReadImage(os.path.join(os.path.join(raw_dataset_path ,'data'), ct_file), sitk.sitkInt16)
    #转换成 numpy格式
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(os.path.join(raw_dataset_path ,'label'), ct_file.replace('volume', 'segmentation')),
                            sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    print("裁剪前:{}".format(ct.GetSize(), seg.GetSize()))
    

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 找到肝脏区域开始和结束的slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())

    new_seg = sitk.GetImageFromArray(seg_array)
    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    print("裁剪后:{}".format(new_ct.GetSize(), new_seg.GetSize()))

    sitk.WriteImage(new_ct, os.path.join(os.path.join(fixed_dataset_path ,'data'), ct_file))
    sitk.WriteImage(new_seg,
                    os.path.join(os.path.join(fixed_dataset_path , 'label'), ct_file.replace('volume', 'segmentation')))    # 替换名字

# 3. 把nii的数据保存为jpg格式
data_path = 'work/new_traindata/data'
label_path = 'work/new_traindata/label'    # 存放的是分割图像
count = 0
if not os.path.exists('/home/aistudio/work/newdata'):
    os.mkdir('/home/aistudio/work/newdata')
    os.makedirs(os.path.join('/home/aistudio/work/newdata','origin'))
    os.makedirs(os.path.join('/home/aistudio/work/newdata','label'))
for f in os.listdir(data_path):
    origin_path= os.path.join(data_path, f)
    seg_path = os.path.join(label_path,f).replace('volume','segmentation')   # 替换名字
    origin_array = sitk.GetArrayFromImage(sitk.ReadImage(origin_path))
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    for i in range(seg_array.shape[0]):
        seg_image = seg_array[i,:,:]   # 取第i张分割好的图像数据   
        seg_image = np.rot90(np.transpose(seg_image, (1,0)))   # #将矩阵img转置  后  再逆时针旋转90°    
        origin_image = origin_array[i,:,:]    # 取第i张原始没分割的图像数据 
        origin_image = np.rot90(np.transpose(origin_image, (1,0)))   # #将矩阵img转置  后  再逆时针旋转90°    
        cv2.imwrite('work/newdata/label/'+str(count) + '.png', seg_image)
        cv2.imwrite('work/newdata/origin/'+str(count) + '.jpg', origin_image)
        count += 1

print(count)

image = cv2.imread('work/newdata/origin/51.jpg',0)
label = cv2.imread('work/newdata/label/51.png',0)
plt.figure(figsize=(10,5))   #figsize:指定figure的宽和高，单位为英寸
plt.subplot(121)   # (nrows, ncols, index, **kwargs)
plt.imshow(image,'gray')    # 第二个参数是显示的模式

plt.subplot(122)
plt.imshow(label, 'gray')
plt.show()

# 4.创建 train.txt, val.txt, test.txt文档
random.seed(2021)    # 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同
path_origin = '/home/aistudio/work/newdata/origin'
path_label = '/home/aistudio/work/newdata/label'
files = list(filter(lambda x: x.endswith('.jpg'), os.listdir(path_origin)))   # 获取到path_origin 下所有 以jpg结尾的文件名列表。
random.shuffle(files)
rate = int(len(files) * 0.8)   #训练集和测试集8：2
train_txt = open('/home/aistudio/work/newdata/train_list.txt','w')   # 以w方式打开，不能读出。w+可读写
val_txt = open('/home/aistudio/work/newdata/val_list.txt','w')
test_txt = open('/home/aistudio/work/newdata/test_list.txt','w')
for i,f in enumerate(files):
    image_path = os.path.join(path_origin, f)
    label_name = f.split('.')[0]+ '.png'
    label_path = os.path.join(path_label, label_name)
    if i < rate:
        train_txt.write(image_path + ' ' + label_path+ '\n')
    else:
        if i%2 :
            val_txt.write(image_path + ' ' + label_path+ '\n')
        else:
            test_txt.write(image_path + ' ' + label_path+ '\n')
train_txt.close()
val_txt.close()
test_txt.close()
print('完成')

# 5.创建Transform 和DataSet
import paddleseg.transforms as T
from paddleseg.datasets import OpticDiscSeg,Dataset

train_transforms = [
    #补全  
    # 这里可以加数据增强，例如水平翻转(RandomHorizontalFlip())、随机旋转(RandomRotation())、随机缩放(RandomScaleAspect)等
    # 
    T.RandomHorizontalFlip(),                                                              # 水平翻转
    T.RandomScaleAspect(min_scale = 0.8, aspect_ratio = 0.5),                              # 随机缩放
    T.RandomDistort(),                                                                     # 随机扭曲
    T.RandomBlur(),                                                                        # 随机模糊
    T.RandomRotation(max_rotation = 10,im_padding_value =(0,0,0),label_padding_value = 0), # 随机旋转
    T.Resize(target_size=(512,512)),
    T.Normalize()  
]
val_transforms = [
    #补全
    T.Resize(target_size=(512,512)),
    T.Normalize()
]
test_transforms = [
    T.Resize(target_size=(512,512)),
    T.Normalize()
]

dataset_root = '/home/aistudio/work/newdata'
train_path  = '/home/aistudio/work/newdata/train_list.txt'
val_path  = '/home/aistudio/work/newdata/val_list.txt'
test_path  = '/home/aistudio/work/newdata/test_list.txt'
# 构建训练集
train_dataset = Dataset(
    #补全
    dataset_root = dataset_root,
    train_path = train_path,
    transforms = train_transforms,
    num_classes = 3,
    mode = 'train'
                  )
#验证集
val_dataset = Dataset(
    #补全
    dataset_root =dataset_root,
    val_path = val_path,
    num_classes = 3,
    transforms = val_transforms,
    mode = 'val'
                  )

#测试集
test_dataset = Dataset(
    #补全
    dataset_root = dataset_root,
    test_path = test_path,
    num_classes = 3,
    transforms = test_transforms,
    mode = 'test'           
                  )


#预览数据
#没有显示 ，在运行一次
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(16,16))
for i in range(1,6,2):
    img, label = train_dataset[131]
    img = np.transpose(img, (1,2,0))
    img = img*0.5 + 0.5
    plt.subplot(3,2,i),plt.imshow(img,'gray'),plt.title('img'),plt.xticks([]),plt.yticks([])
    plt.subplot(3,2,i+1),plt.imshow(label,'gray'),plt.title('label'),plt.xticks([]),plt.yticks([])
    plt.show

# 6.设置网络和 损失函数、优化器等
#补全 导入模型
import paddle
from paddleseg.models import UNet

# 这里可以调用其他分割网络，具体可以去paddleseg.models里查看
model = UNet(num_classes = 3)

from paddleseg.models.losses import CrossEntropyLoss,DiceLoss
import paddle
# 设置学习率  
base_lr = 0.01
#自己换学习率和优化器，能不能上高分
lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=1800, verbose=False)
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
losses = {}
#自己尝试组合dice损失函数，会不会效果更好
losses['types'] = [CrossEntropyLoss()] 
losses['coef'] = [1]

# 7.开始训练
from paddleseg.core import train
#查看文档自己补全
train(
    #补全
    model = model,                   # 训练模型
    train_dataset = train_dataset,   # 训练集
    val_dataset = val_dataset,       # 验证集
    optimizer = optimizer,           # 优化器
    losses = losses,                 # 损失函数
    save_dir = 'my_save_model',         # 训练模型保存路径
    iters = 2500,                    # 总迭代次数
    batch_size = 2,                  # 每批处理图片张数
    save_interval = 250,             # 保存和评估间隔
    log_iters = 20,                  # 训练日志打印间隔
    use_vdl = True                   # 使用VisualDL
    )

# 8.0评估测试集
from paddleseg.core import evaluate
model = UNet(num_classes=3)
#换自己保存的模型文件
model_path = '/home/aistudio/my_save_model/best_model/model.pdparams'
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)
evaluate(model,val_dataset)

from paddleseg.core import predict
transforms = T.Compose([
    T.Resize(target_size=(512, 512)),
    T.Normalize()
])

model = UNet(num_classes=3)
#生成图片列表
image_list = []
with open('/home/aistudio/work/newdata/test_list.txt' ,'r') as f:
    for line in f.readlines():
        image_list.append(line.split()[0])

predict(
        model,
        #换自己保存的模型文件
        model_path = '/home/aistudio/my_save_model/best_model/model.pdparams',
        transforms=transforms,
        image_list=image_list,
        save_dir='/home/aistudio/save_model/results',
    )

# 9.预览分割结果
num = 6
img_list = random.sample(image_list, num)
pre_path = 'save_model/results/pseudo_color_prediction'
plt.figure(figsize=(12,num*4))
index = 1
for i in range(len(img_list)):
    plt.subplot(num,3,index)
    img_origin = cv2.imread(img_list[i],0)
    plt.title('origin')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_origin,'gray')

    plt.subplot(num,3,index+1)
    label_path = (img_list[i].replace('origin', 'label')).replace('jpg','png')
    img_label = cv2.imread(label_path,0)
    plt.title('label')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_label, 'gray')

    plt.subplot(num,3,index+2)
    predict_path = os.path.join(pre_path, os.path.basename(label_path))
    img_pre = cv2.imread(predict_path,0)
    plt.title('predict')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_pre, 'gray')

    index += 3

plt.show()
