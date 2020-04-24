import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

img = Image.open(r'C:\Users\MR  li\Pictures\person.jpg')#r 是转义符，Windows系统文件地址和pycharm不一样

transform = transforms.Compose([transforms.ToTensor()])#totensor 得到（C*H*W)
img_tensor = transform(img)#[3 128 64] 得到tensor形式

'''方法一
image.show()方式，来源于PIL
'''
to_pil_img = transforms.ToPILImage()#tensor 重新转化成图片格式
img2 = to_pil_img(img_tensor)

#img2.show()

'''方法二
plt.imshow()  plt.show()  针对的是数组形式 （array）
所以需要把tensor 转化成numpy数组形式
'''
imgarray = img_tensor.numpy()#3 128 64
img3 = np.transpose(imgarray,(1,2,0))# 128 64 3  plt需要数组格式为  H W C

plt.suptitle("orignal -PIL- again")
plt.subplot(1,3,1)
plt.title("orignal img")
plt.imshow(img)#显示输入图片

plt.subplot(1,3,2)
plt.title("PIL img")
plt.imshow(img2)


plt.subplot(1,3,3)
plt.title("again img")
plt.imshow(img3)#显示重新转换回来
plt.show()