import os
import numpy as np
import cv2
from tqdm import tqdm

# concat_contrast_D3_256:
# path_list = ['contrast_agis', 'test_1125', 'test_local_3style_newdis_w420_1125']
# concat_contrast_D123:
# path_list = ['contrast_agis_80', 'test_1125', 'test_style_num3_newdis_w420_1125']
# concat_multistyle:
# path_list = ['test', 'test_style_num2', 'test_style_num3', 'test_style_num4']
# concat_newdis:
# path_list = ['test', 'test_newdis_w400', 'test_newdis_w420', 'test_newdis_w450',
#     'test_newdis_w500', 'test_newdis_w600']
# concat_ms+dis:
# path_list = ['test', 'test_style_num3', 'test_newdis_w420', 'test_style_num3_newdis_w420']
data_path = './243'
result_path = './concat_contrast_D123'
path_list = ['contrast_agis_80', 'test_1125', 'test_style_num3_newdis_w420_1125']

if not os.path.exists(result_path):
    os.mkdir(result_path)

resol = 80
def concat_image(dn, img_n):
    target_img = cv2.imread(os.path.join(data_path, path_list[0], 
        'D%d/%s_target.png' % (dn, img_n)))
    row = 5
    col = 10

    # print('step 1...')
    if resol == 256:
        target_img = cv2.resize(target_img, (resol * col, resol * row))

    init_imgs = [target_img]
    num_256 = 0
    for i in range(len(path_list)):
        img = cv2.imread(os.path.join(data_path, path_list[i], 
            'D%d/%s_output.png' % (dn, img_n)))
        h, w, c = img.shape
        if resol == 256 and w == resol * col:
            # if num_256 == 0:
            #     img = cv2.imread(os.path.join(data_path, path_list[i], 
            #         'D%d/%s_target.png' % (dn, img_n)))
            #     init_imgs.append(img)
            #     num_256 += 1
            num_256 += 1
        if resol == 256 and w == 80 * col:
            new_row = int(h / 80)
            img = cv2.resize(img, (resol * col, resol * new_row))
        init_imgs.append(img)
    
    # if resol == 256:
    #     img = cv2.imread(os.path.join(data_path, path_list[-1], 
    #         'D%d/%s_target.png' % (dn, img_n)))
    #     init_imgs.append(img)
    #     num_256 += 1

    h, w, c = target_img.shape
    concat_imgs = []
    # print('step 2...')
    for n in range(row * col):
        i = n // col
        j = n % col
        img_list = []
        for k in range(len(init_imgs)):
            img = init_imgs[k][resol*i: resol*(i+1), resol*j: resol*(j+1)]
            if resol == 256 and k == len(init_imgs) - num_256:
                zero_img = np.zeros((resol, 10, c))
                # img_list.append(zero_img)
            img_list.append(img)
        concat_img = np.concatenate(img_list, 1)
        concat_imgs.append(concat_img)

    row = 25
    col = 2
    # print('step 3...')
    row_list = []
    for i in range(row):
        col_list = []
        for j in range(col):
            col_list.append(concat_imgs[i * col + j])
        row_img = np.concatenate(col_list, 1)
        
        row_list.append(row_img)
    result = np.concatenate(row_list, 0)
    save_path = os.path.join(result_path, 'D%d/%s.png' % (dn, img_n))
    cv2.imwrite(save_path, result)


for i in [2]:
    print('processing D%d...' % i)
    path = os.path.join(result_path, 'D%d' % i)
    if not os.path.exists(path):
        os.mkdir(path)
    L = os.listdir(os.path.join(data_path, path_list[0], 'D%d' % i))
    L.sort()
    for img_name in tqdm(L):
        if img_name[-10:-4] == 'output':
            concat_image(i, img_name[:-11])