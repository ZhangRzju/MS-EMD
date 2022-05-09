import pickle
import os
import random
from tqdm import tqdm
from src import utils

font_list = utils.load_list('./datasets/243/font_list.txt')
char_list = utils.load_list('./datasets/243/char_list.txt')
K1 = int(len(font_list)*0.75)   # 选取font_list中的前K1种字体
K2 = int(len(char_list)*0.75)   # 选取char_list中的前K2个汉字
# 训练/测试/插值的数据量
train_iter = 300000
test_iter = 10000
# interp_iter = 10000
content_sample_num = 10
style_sample_num = 10
base_path = './datasets/243'
image_path = base_path + '/images'
# skel_path = base_path + '/skels'
train_save_path = base_path + '/train'
test_save_path = base_path + '/test'
# interp_save_path = base_path + '/interp'
style_num = 3   # the number of style encoders


def get_data_list(mode, path_n=1, with_skel=False, target_font1=None, 
        target_font2=None, target_fonts=None, target_chars=None):
    if mode == 'train':
        max_iter = train_iter
        savepath = train_save_path
    elif mode == 'test':
        max_iter = test_iter
        savepath = test_save_path + '/D%d' % path_n
    elif mode == 'interp':
        max_iter = interp_iter
        savepath = interp_save_path
    else:
        print('Wrong mode!')
        return
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # font list and char list to sample target image
    if path_n == 1:
        target_font_L = font_list[:K1]
        target_char_L = char_list[:K2]
    elif path_n == 2:
        target_font_L = font_list[:K1]
        target_char_L = char_list[K2:]
    elif path_n == 3:
        target_font_L = font_list[K1:]
        target_char_L = char_list[:K2]
    else:
        target_font_L = font_list[K1:]
        target_char_L = char_list[K2:]
    # font list and char list to sample style and content images
    font_L = font_list[:K1]
    char_L = char_list[:K2]
    if target_fonts != None:
        target_fonts.sort()

    style_list = []
    content_list = []
    target_list = []
    if with_skel:
        skel_list = []
    
    for i in tqdm(range(max_iter)):
        # target sampling
        if target_chars != None:
            target_char = target_chars[i % len(target_chars)]
        else:
            target_char = random.sample(target_char_L, 1)[0]
        
        if target_fonts != None:
            target_font1 = target_fonts[i // len(target_chars)]
        
        if target_font1 != None:
            target_font = target_font1
        else:
            target_font = random.sample(target_font_L, 1)[0]

        sample_path = os.path.join(image_path, target_font, target_char)
        target_list.append(sample_path)

        # with content fixed, sampling style randomly
        temp_L = font_L[:]
        try:
            temp_L.remove(target_font)
        except:
            pass
        fonts = random.sample(temp_L, content_sample_num)

        for font in fonts:
            sample_path = os.path.join(image_path, font, target_char)
            content_list.append(sample_path)
            if with_skel:
                # with content of skel fixed, sampling style randomly
                sample_path = os.path.join(skel_path, font, target_char)
                skel_list.append(sample_path)
        
        # with style fixed, sampling content randomly
        temp_L = char_L[:]
        try:
            temp_L.remove(target_char)
        except:
            pass
        chars = random.sample(temp_L, style_sample_num)

        for char in chars:
            sample_path = os.path.join(image_path, target_font, char)
            style_list.append(sample_path)
        
        # extra data for interpolating
        if mode == 'interp':
            if target_font2 != None:
                target_font = target_font2
            else:
                temp_L = target_font_L[:]
                temp_L.remove(target_font)
                target_font = random.sample(temp_L, 1)[0]
            sample_path = os.path.join(image_path, target_font, target_char)
            target_list.append(sample_path)

            for char in chars:
                sample_path = os.path.join(image_path, target_font, char)
                style_list.append(sample_path)

    with open(savepath + '/style.pkl', 'wb') as f:
        pickle.dump(style_list, f)
    with open(savepath + '/content.pkl', 'wb') as f:
        pickle.dump(content_list, f)
    with open(savepath + '/target.pkl', 'wb') as f:
        pickle.dump(target_list, f)

    if with_skel:
        with open(savepath + '/skel.pkl', 'wb') as f:
            pickle.dump(skel_list, f)


def get_multi_style_data_list(mode, style_n=3, path_n=1, target_font1=None, target_font2=None,
        target_font3=None, target_chars=None):
    if mode == 'train':
        max_iter = train_iter
        savepath = train_save_path
    elif mode == 'test':
        max_iter = test_iter
        savepath = test_save_path + '/D%d' % path_n
    else:
        print('Wrong mode!')
        return
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # font list and char list of sampling target
    if path_n == 1:
        target_font_L = font_list[:K1]
        target_char_L = char_list[:K2]
    elif path_n == 2:
        target_font_L = font_list[:K1]
        target_char_L = char_list[K2:]
    elif path_n == 3:
        target_font_L = font_list[K1:]
        target_char_L = char_list[:K2]
    else:
        target_font_L = font_list[K1:]
        target_char_L = char_list[K2:]
    # font list and char list of sampling style and content
    font_L = font_list[:K1]
    char_L = char_list[:K2]

    style_list = []
    content_list = []
    target_list = []
    
    for i in tqdm(range(max_iter)):
        if target_chars != None:
            target_char = target_chars[i % len(target_chars)]
        else:
            target_char = random.sample(target_char_L, 1)[0]
        # sampling content randomly
        temp_L = char_L[:]
        try:
            temp_L.remove(target_char)
        except:
            pass
        chars = random.sample(temp_L, style_sample_num)

        exists_target_fonts = []
        for k in range(style_n):
            # target sampling
            if k == 0 and target_font1 != None:
                target_font = target_font1
            elif k == 1 and target_font2 != None:
                target_font = target_font2
            elif k == 2 and target_font3 != None:
                target_font = target_font3
            else:
                temp_L = target_font_L[:]
                for font in exists_target_fonts:
                    temp_L.remove(font)
                target_font = random.sample(temp_L, 1)[0]
            exists_target_fonts.append(target_font)
            sample_path = os.path.join(image_path, target_font, target_char)
            target_list.append(sample_path)
            # with style fixed, sampling content
            for char in chars:
                sample_path = os.path.join(image_path, target_font, char)
                style_list.append(sample_path)

        # with content fixed, sampling style randomly
        temp_L = font_L[:]
        for font in exists_target_fonts:
            try:
                temp_L.remove(font)
            except:
                pass
        fonts = random.sample(temp_L, content_sample_num)

        for font in fonts:
            sample_path = os.path.join(image_path, font, target_char)
            content_list.append(sample_path)

    with open(savepath + '/style.pkl', 'wb') as f:
        pickle.dump(style_list, f)
    with open(savepath + '/content.pkl', 'wb') as f:
        pickle.dump(content_list, f)
    with open(savepath + '/target.pkl', 'wb') as f:
        pickle.dump(target_list, f)


# get_data_list('train', with_skel=True)
# get_data_list('test', 1, target_fonts=font_list[:K1], target_chars=char_list[:K2])
# get_data_list('test', 2, target_fonts=font_list[:K1], target_chars=char_list[K2:])
# get_data_list('test', 3, target_fonts=font_list[K1:], target_chars=char_list[:K2])
# get_data_list('test', 4, target_fonts=font_list[K1:], target_chars=char_list[K2:])

# get_data_list('interp', with_skel=True)


get_multi_style_data_list('train', style_num)
get_multi_style_data_list('test', style_num, 1)
get_multi_style_data_list('test', style_num, 2)
get_multi_style_data_list('test', style_num, 3)
get_multi_style_data_list('test', style_num, 4)