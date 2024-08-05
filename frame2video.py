import cv2
import os
 
#图像文件夹路径
folder_path = 'vis/elephant'#换成自己图像的绝对路径
 
#获取.jpg格式的图像文件列表
image_files = sorted(os.listdir(folder_path))
 
#获取第一张图像的尺寸
image_path = os.path.join(folder_path,image_files[0])
first_image = cv2.imread(image_path)
height,width,channels = first_image.shape
 
#设置视频输出路径和相关参数
output_path = 'vis/elephant.mp4'
fps = 30.0 #帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #编码器
 
#创建视频写入器对象
video_writer = cv2.VideoWriter(output_path,fourcc,fps,(width,height))
 
#把图像逐帧写入视频
for image_file in image_files:
    image_path = os.path.join(folder_path,image_file)
    image = cv2.imread(image_path)
    video_writer.write(image)
 
#释放资源
video_writer.release()