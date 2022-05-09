import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np 
import pygame
import random

# 将宽度和高度不等的图片填充成宽度和高度相等的图片
def img_resize(img, size=80):
    h, w = img.shape
    if h > w:       # 横向填充
        k = h - w
        pad_img1 = np.zeros((h, k//2))
        pad_img1[:, :] = 255
        pad_img2 = np.zeros((h, k - k//2))
        pad_img2[:, :] = 255
        new_img = np.concatenate([pad_img1, img, pad_img2], axis=1)
    else:           # 纵向填充
        k = w - h
        pad_img1 = np.zeros((k//2, w))
        pad_img1[:, :] = 255
        pad_img2 = np.zeros((k - k//2, w))
        pad_img2[:, :] = 255
        new_img = np.concatenate([pad_img1, img, pad_img2], axis=0)

    new_img = cv2.resize(new_img, (size, size))
    return new_img

def font2img(img_path, font_path, resol):
    print("initializing image folder...")
    if os.path.exists(img_path):
        # 清空img_path
        for f in os.listdir(img_path):
            filepath = img_path + '/' + f
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath, True)
    else:
        os.makedirs(img_path)
    print('ok.')
    
    # 读取图片的索引
    char_list = []
    text_path = './datasets/chars_1500.txt'
    with open(text_path, encoding='utf8') as f:
        for line in f:
            for char in line:
                if char != '\n':
                    char_list.append(char)

    font_list = os.listdir(font_path)
    font_list.sort()
    print(font_list)
    try:
        print("Tencent font:", font_list.index("TencentSansUprightW3.otf"))
    except Exception as e:
        print(e)

    pygame.init()
    for i in tqdm(range(len(font_list))):
        font = pygame.font.Font(font_path + '/' + font_list[i], resol)
        path = img_path + '/%03d' % i
        if not os.path.exists(path):
            os.mkdir(path)
        
        for j in range(len(char_list)):
            text = char_list[j]
            # 渲染图片, 设置背景颜色和字体样式, 前面的颜色是字体颜色
            ftext = font.render(text, True, (0, 0, 0), (255, 255, 255))
            img = pygame.surfarray.array3d(ftext)
            # 如果不转置，cv2保存的图片就是相反的
            img = np.transpose(img, [1, 0, 2])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(path + '/%04d.png' % j, img_resize(img, resol))


def shuffle(base_path, img_path):
    font_L = os.listdir(img_path)
    char_L = os.listdir(os.path.join(img_path, font_L[0]))
    random.shuffle(font_L)
    random.shuffle(char_L)

    with open(base_path + '/font_list.txt', 'w', encoding='utf8') as f1:
        for font in font_L:
            f1.write(font + '\n')
        
    with open(base_path + '/char_list.txt', 'w', encoding='utf8') as f2:
        for char in char_L:
            f2.write(char + '\n')

base_path = './datasets/243'
img_path = base_path + '/images'
font_path = './fonts/fonts_243'
# resolution of images
resol = 80
font2img(img_path, font_path, resol)
shuffle(base_path, img_path)