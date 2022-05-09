import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm, trange
import functools
import shutil

from src import utils

################################################################################
# tencent font separating and labeling

# data_dir = 'results/243/interp_tencent_other/076_094/D1'
# save_dir = 'results/243/interp_tencent_other/076_094/separate'

# if os.path.exists(save_dir):
#     utils.clear_dir(save_dir)
# else:
#     os.mkdir(save_dir)

# with open('datasets/243/char_label.txt', 'r') as f:
#     labels = [line.strip().split(',')[1] for line in f]

# L = os.listdir(data_dir)
# L.sort()
# char_num = 5
# resol = 256
# interp_num = 8
# for i in tqdm(range(len(L))):
#     img = cv2.imread(os.path.join(data_dir, L[i]))
#     for j in range(char_num):
#         c = labels[i*char_num+j]
#         char_unicode = ord(c)
#         for k in range(0, interp_num + 1):
#             if k == 0:
#                 subimg = img[j*resol : (j+1)*resol, :resol]
#             elif k == interp_num:
#                 subimg = img[j*resol : (j+1)*resol, (k+2)*resol : (k+3)*resol]
#             else:
#                 subimg = img[j*resol : (j+1)*resol, (k+1)*resol : (k+2)*resol]
#             # cv2.imwrite(os.path.join(save_dir, '%s%d_%d.png' % (c, char_unicode, k-1)), subimg)
#             cv2.imwrite(os.path.join(save_dir, '%d_%d.png' % (char_unicode, k)), subimg)


#################################################################################
# change image path to 'images_resol_256'

# dir1 = 'datasets/243/interp_3style_tencent'
# dir2 = 'datasets/243/interp_256_3style_tencent'

# def replace_256(dn, src_dir, dst_dir):
#     pkl_list = ['content.pkl', 'style.pkl', 'target.pkl']
#     for pkl_file in pkl_list:
#         with open(os.path.join(src_dir, pkl_file), 'rb') as f:
#             path_L = pickle.load(f)
#         new_path_L = [path.replace('images', 'images_resol_256') for path in path_L]
#         with open(os.path.join(dst_dir, pkl_file), 'wb') as f:
#             pickle.dump(new_path_L, f)

# for dn in range(1, 5):
#     src_dir = os.path.join(dir1, 'D%d' % dn)
#     dst_dir = os.path.join(dir2, 'D%d' % dn)
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
#     replace_256(dn, src_dir, dst_dir)


#################################################################################
# tencent font averaging interpoltion

# img_interval = (0, 5)
# resol = 80
# init_interp_num = 100
# final_interp_num = 12
# data_dir = 'results/243/interp_3style_tencent_num100/D1'
# save_dir = 'results/243/interp_3style_tencent_num100/avg_interp_%d' % final_interp_num
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# L = os.listdir(data_dir)
# L.sort()

# def compute_black_pixels(img):
#     return np.where(img < 127)[0].shape[0]

# def cmp_func(tuple_x, tuple_y):
#     return tuple_x[0] < tuple_y[0]

# f = open('results/243/interp_3style_tencent_num100/pixels.csv', 'w')
# alphas = np.arange(1 / final_interp_num, 1, 1 / final_interp_num)
# interp_img_list = []
# print('averaging interpolation...')
# for i in tqdm(range(len(L))): 
#     img0 = cv2.imread(os.path.join(data_dir, L[i]))
#     for j in range(img_interval[0], img_interval[1]):
#         img = img0[j*resol : (j+1)*resol, :, :]
#         left_img = img[:, : resol, :]
#         gen_left_img = img[:, resol : 2*resol, :]
#         gen_right_img = img[:, (init_interp_num+1)*resol : (init_interp_num+2)*resol, :]
#         right_img = img[:, (init_interp_num+2)*resol :, :]
#         gen_left_pixels = compute_black_pixels(gen_left_img)
#         gen_right_pixels = compute_black_pixels(gen_right_img)
#         subimgs = [img[:, k*resol : (k+1)*resol, :] for k in range(2, init_interp_num + 1)]
#         pixels_list = [(compute_black_pixels(subimgs[k]), k) for k in range(len(subimgs))]

#         for k in range(len(pixels_list)):
#             f.write('%04d,%03d,' % (i*5+j, k) + str(pixels_list[k][0]) + '\n')
#         pixels_list.sort(key=functools.cmp_to_key(cmp_func))
        
#         k = 0
#         img_list = [left_img]
#         for alpha in alphas:
#             pixels = (1 - alpha) * gen_left_pixels + alpha * gen_right_pixels
#             while k < len(pixels_list):
#                 if gen_left_pixels < gen_right_pixels and pixels_list[k][0] > pixels:
#                     img_list.append(subimgs[pixels_list[k][1]])
#                     k += 1
#                     break
#                 elif gen_left_pixels > gen_right_pixels and pixels_list[k][0] < pixels:
#                     img_list.append(subimgs[pixels_list[k][1]])
#                     k += 1
#                     break
#                 else:
#                     k += 1
#         img_list.append(right_img)
#         interp_img_list.append(np.concatenate(img_list, 1))

# print('concatenating...')
# row, col = 5, 1
# save_num_per_img = row * col
# assert len(interp_img_list) % (save_num_per_img) == 0
# for i in range(int(len(interp_img_list)/save_num_per_img)):
#     concat_row_list = []
#     for j in range(row):
#         concat_col_list = [interp_img_list[i*save_num_per_img+j*col+k] for k in range(col)]
#         concat_row_img = np.concatenate(concat_col_list, 1)
#         concat_row_list.append(concat_row_img)
#     concat_img = np.concatenate(concat_row_list, 0)
#     cv2.imwrite(os.path.join(save_dir, '%05d.png' % i), concat_img)


################################################################################
# tencent font affine transformation

# data_dir = 'results/243/interp_local_3style_tencent_num100/separate'
# save_dir = 'results/243/interp_local_3style_tencent_num100/affine'

# if os.path.exists(save_dir):
#     utils.clear_dir(save_dir)
# else:
#     os.mkdir(save_dir)

# L = os.listdir(data_dir)
# L.sort()

# pos1 = np.float32([[0, 0], [256, 0], [0, 128]])
# pos2 = np.float32([[18, 0], [274, 0], [0, 128]])
# M = cv2.getAffineTransform(pos1, pos2)

# def affine(path):
#     img = cv2.imread(os.path.join(data_dir, path), 0)
#     row, col = img.shape
#     img_aff = cv2.warpAffine(img, M, (row, col))
#     # print(img_aff[0, :])# 18
#     # print(img_aff[:, 0])# 128
#     # print(img_aff[-1, :])# 18
#     # print(img_aff[:, -1])# 127
#     img_aff[:128, :18] = 255
#     img_aff[-128:, -18:] = 255
#     return img_aff

# for i in tqdm(range(len(L))):
#     img_affine = affine(L[i])
#     cv2.imwrite(os.path.join(save_dir, L[i]), img_affine)


################################################################################
# concatenate few interpolation results, Hei Kai Li Song

# data_dirs = ['results/243/interp_HKLS/HKL', 'results/243/interp_HKLS/LKS', 'results/243/interp_HKLS/SHK']
# save_dir = 'results/243/interp_HKLS/concat'
# if os.path.exists(save_dir):
#     utils.clear_dir(save_dir)
# else:
#     os.mkdir(save_dir)

# L = os.listdir(data_dirs[0] + '/D1')
# L.sort()
# char_per_img = 5
# resol = 256

# for i in tqdm(range(len(L))):
#     img_list = []
#     for data_dir in data_dirs:
#         img = cv2.imread(os.path.join(data_dir, 'D1', L[i]), 0)
#         img_list.append(img[:2*resol*char_per_img, :])
#     concat_img = np.concatenate(img_list, 0)
#     cv2.imwrite(os.path.join(save_dir, L[i]), concat_img)


################################################################################
# find position of certain font in font list

# data_dir = 'fonts/fonts_243'
# L = os.listdir(data_dir)
# L.sort()

# font_L = ['FZSTJW.TTF']

# for font_name in font_L:
#     print(font_name, L.index(font_name))


#################################################################################
# prepare data for zi2zi

# source_dir = 'fonts/siyuan_song_imgs_80/000'
# data_dir = 'datasets/243/images'
# font_list = utils.load_list('datasets/243/font_list.txt')
# char_list = utils.load_list('datasets/243/char_list.txt')
# len1 = int(0.75 * len(font_list))
# len2 = int(0.75 * len(char_list))
# # train_list = font_list[:len1]
# # indices = [181, 50, 129, 201, 149, 106, 35, 187, 72, 77, 15, 19, 30, 59, 80, 67, 68]
# # for i in range(len(indices)):
# #     print(train_list.index('%03d' % indices[i]), end=', ')

# save_dir = '/mnt/nfs-shared/ziti/zi2zi/datasets/image_emd_train_80resize'
# if os.path.exists(save_dir):
#     utils.clear_dir(save_dir)
# else:
#     os.makedirs(save_dir)
# for i in tqdm(range(len1)):
#     for j in range(len2):
#         source_img = cv2.imread(os.path.join(source_dir, char_list[j]))
#         target_img = cv2.imread(os.path.join(data_dir, font_list[i], char_list[j]))
#         source_img = cv2.resize(source_img, (256, 256))
#         target_img = cv2.resize(target_img, (256, 256))
#         concat_img = np.concatenate([target_img, source_img], 1)
#         cv2.imwrite(os.path.join(save_dir, '%03d_%04d.jpg' % (i, j)), concat_img)

# save_dir = '/mnt/nfs-shared/ziti/zi2zi/datasets/image_emd_val_80resize'
# if os.path.exists(save_dir):
#     utils.clear_dir(save_dir)
# else:
#     os.makedirs(save_dir)
# for i in tqdm(range(len1)):
#     for j in range(len2, len(char_list)):
#         source_img = cv2.imread(os.path.join(source_dir, char_list[j]))
#         target_img = cv2.imread(os.path.join(data_dir, font_list[i], char_list[j]))
#         source_img = cv2.resize(source_img, (256, 256))
#         target_img = cv2.resize(target_img, (256, 256))
#         concat_img = np.concatenate([target_img, source_img], 1)
#         cv2.imwrite(os.path.join(save_dir, '%03d_%04d.jpg' % (i, j)), concat_img)

################################################################################
# resize test results to 256

# output_img_dir = 'results/243/test'
# target_img_dir = 'results/243/test_old_256'
# dst_dir = 'results/243/test_resize256'

# for i in range(1, 5):
#     L = os.listdir(os.path.join(output_img_dir, 'D%d' % i))
#     save_dir = os.path.join(dst_dir, 'D%d' % i)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for j in tqdm(range(len(L))):
#         if 'target' in L[j]:
#             img = cv2.imread(os.path.join(target_img_dir, 'D%d' % i, L[j]), 0)
#         else:
#             img = cv2.imread(os.path.join(output_img_dir, 'D%d' % i, L[j]), 0)
#             row, col = img.shape[0] // 80, img.shape[1] // 80
#             img = cv2.resize(img, (col*256, row*256))
#         cv2.imwrite(os.path.join(save_dir, L[j]), img)


################################################################################
# compare interpolation results between original emd and local

# # compare_list = ['SH', 'SK', 'HK']
# compare_list = ['OL']
# origin_dir = '/data/wuyonglin/EMD_new/results/243/interp_HKLS'
# # new_dir = 'results/243/interp_HKLS/SHK/D1'
# new_dir = 'results/243/interp_HKLS/OLF/D1'
# save_dir = 'results/243/interp_HKLS/compare'
# char_per_img = 5
# resol = 256

# for i in range(len(compare_list)):
#     if not os.path.exists(os.path.join(save_dir, compare_list[i])):
#         os.makedirs(os.path.join(save_dir, compare_list[i]))
#     L = os.listdir(origin_dir + '/%s' % compare_list[i])
#     for j in tqdm(range(len(L))):
#         origin_img = cv2.imread(os.path.join(origin_dir, compare_list[i], L[j]), 0)
#         row, col = origin_img.shape[0] // 80, origin_img.shape[1] // 80
#         origin_img = cv2.resize(origin_img, (col * resol, row * resol))
#         new_img = cv2.imread(os.path.join(new_dir, L[j]), 0)[i*resol*char_per_img : (i+1)*resol*char_per_img]
#         concat_list = []
#         for k in range(char_per_img):
#             concat_list.append(origin_img[k*resol : (k+1)*resol, :])
#             concat_list.append(new_img[k*resol : (k+1)*resol, :])
#         cv2.imwrite(os.path.join(save_dir, compare_list[i], L[j]), np.concatenate(concat_list, 0))


################################################################################
# prepare data for agis-net

# A_dir = '/mnt/nfs-shared/wuyonglin/AGIS-Net/datasets/average_skeleton'
# B_dir = 'datasets/243/images_resol_256'
# save_dir = '/mnt/nfs-shared/wuyonglin/AGIS-Net/datasets/skeleton_gray_color'
# style_L = [172, 370, 222,  37, 220, 317, 333, 494, 468,  25,
#             440, 208, 488, 177, 167, 104, 430, 383, 422, 174,
#             441, 475, 473,  72,   9, 389, 132, 412,  24, 288,
#             453, 372, 181, 322, 115,  34, 345, 243, 188, 118,
#             142, 197, 429, 358, 223, 121,  20, 241, 178, 238,
#             272, 182, 384, 295, 490,  98,  96, 476, 226, 129,
#             305,  28, 207, 351, 193, 378, 390, 353, 452, 240,
#             477, 214, 306, 373,  63, 248, 323, 109,  21, 381,
#             393, 263, 111,  92, 231, 114, 218,  69, 482, 252,
#             257, 300, 283, 420,  62, 154, 146, 478,  89, 419]
# path_L = ['D3']
# for path in path_L:
#     if os.path.exists(os.path.join(save_dir, path)):
#         utils.clear_dir(os.path.join(save_dir, path))
#     else:
#         os.makedirs(os.path.join(save_dir, path))

# font_list = utils.load_list('datasets/243/font_list.txt')
# B_list = utils.load_list('datasets/243/char_label.txt')
# len1 = int(0.75 * len(font_list))
# len2 = int(0.75 * len(B_list))

# path = 'train'
# print('processing %s images...' % path)
# for i in tqdm(range(len1)):
#     for j in range(len2):
#         img_name, c, gb2312_idx = B_list[j].split(',')
#         A_img = cv2.imread(os.path.join(A_dir, '%s.png' % gb2312_idx))
#         A_img = cv2.resize(A_img, (64, 64))
#         B_img = cv2.imread(os.path.join(B_dir, font_list[i], img_name))
#         B_img = cv2.resize(B_img, (64, 64))
#         concat_img = np.concatenate([A_img, B_img, B_img], 1)
#         cv2.imwrite(os.path.join(save_dir, path, '%s_%04d.png' % (font_list[i], j)), concat_img)

# path = 'val'
# print('processing %s images...' % path)
# for i in tqdm(range(len1)):
#     for j in range(len2, len(B_list)):
#         img_name, c, gb2312_idx = B_list[j].split(',')
#         A_img = cv2.imread(os.path.join(A_dir, '%s.png' % gb2312_idx))
#         A_img = cv2.resize(A_img, (64, 64))
#         B_img = cv2.imread(os.path.join(B_dir, font_list[i], img_name))
#         B_img = cv2.resize(B_img, (64, 64))
#         concat_img = np.concatenate([A_img, B_img, B_img], 1)
#         cv2.imwrite(os.path.join(save_dir, path, '%s_%04d.png' % (font_list[i], j)), concat_img)

# path = 'D3'
# print('processing %s images...' % path)
# for i in tqdm(range(len1, len(font_list))):
#     for j in range(len2):
#         img_name, c, gb2312_idx = B_list[j].split(',')
#         A_img = cv2.imread(os.path.join(A_dir, '%s.png' % gb2312_idx))
#         A_img = cv2.resize(A_img, (64, 64))
#         B_img = cv2.imread(os.path.join(B_dir, font_list[i], img_name))
#         B_img = cv2.resize(B_img, (64, 64))
#         concat_img = np.concatenate([A_img, B_img, B_img], 1)
#         cv2.imwrite(os.path.join(save_dir, path, '%s_%04d.png' % (font_list[i], j)), concat_img)

# path = 'style'
# print('processing %s images...' % path)
# for i in tqdm(range(len1, len(font_list))):
#     for j in range(len(style_L)):
#         src_path = os.path.join(save_dir, 'D3/%s_%04d.png' % (font_list[i], style_L[j]))
#         dst_path = os.path.join(save_dir, 'style/%s_%04d.png' % (font_list[i], style_L[j]))
#         shutil.copy(src_path, dst_path)


################################################################################
# prepare agis results for quantification

# dn = 3
# resol = 80
# src_dir = '/mnt/nfs-shared/wuyonglin/AGIS-Net/results/skeleton_gray_color/D3/images'
# output_dir = '/mnt/nfs-shared/wuyonglin/EMD_new_tf2/results/243/contrast_agis_shape_80/outputs/D%d' % dn
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# L = os.listdir(src_dir)
# L.sort()

# for i in tqdm(range(len(L)//3)):
#     assert 'fake_B' in L[3*i+1]
#     output_img = cv2.imread(os.path.join(src_dir, L[3*i+1]))
#     output_img = cv2.resize(output_img, (resol, resol))
#     output_path = os.path.join(output_dir, '%s.png' % (L[3*i+1][:8]))
#     cv2.imwrite(output_path, output_img)


# src_dir = 'datasets/243/images'
# dst_dir = 'results/243/contrast_agis_shape_80/targets/D%d' % dn
# if not os.path.exists(dst_dir):
#     os.makedirs(dst_dir)

# font_list = utils.load_list('datasets/243/font_list.txt')
# char_list = utils.load_list('datasets/243/char_list.txt')
# K1 = int(len(font_list)*0.75)
# K2 = int(len(char_list)*0.75)
# if dn == 1:
#     font_L = font_list[:K1]
#     char_L = char_list[:K2]
# elif dn == 2:
#     font_L = font_list[:K1]
#     char_L = char_list[K2:]
# elif dn == 3:
#     font_L = font_list[K1:]
#     char_L = char_list[:K2]
# else:
#     font_L = font_list[K1:]
#     char_L = char_list[K2:]
# for i in tqdm(range(len(font_L))):
#     for j in range(len(char_L)):
#         src_path = os.path.join(src_dir, font_L[i], char_L[j])
#         dst_path = os.path.join(dst_dir, '%s_%04d.png' % (font_L[i], j))
#         shutil.copy(src_path, dst_path)


# # concatenate agis results

# output_dir = 'results/243/contrast_agis_shape_80/outputs/D%d' % dn
# target_dir = 'results/243/contrast_agis_shape_80/targets/D%d' % dn
# save_dir = 'results/243/contrast_agis_shape_80/D%d' % dn
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# L = os.listdir(output_dir)
# L.sort()

# row, col = 5, 10
# for i in tqdm(range(len(L)//(row*col))):
#     output_row_list = []
#     target_row_list = []
#     concat_row_list = []
#     for j in range(row):
#         output_col_list = []
#         target_col_list = []
#         concat_col_list = []
#         for k in range(col):
#             output_col_list.append(cv2.imread(os.path.join(output_dir, L[i*row*col+j*col+k])))
#             target_col_list.append(cv2.imread(os.path.join(target_dir, L[i*row*col+j*col+k])))
#             concat_col_list.append(cv2.imread(os.path.join(target_dir, L[i*row*col+j*col+k])))
#             concat_col_list.append(cv2.imread(os.path.join(output_dir, L[i*row*col+j*col+k])))
#         output_row_list.append(np.concatenate(output_col_list, 1))
#         target_row_list.append(np.concatenate(target_col_list, 1))
#         concat_row_list.append(np.concatenate(concat_col_list, 1))
#     output_img = np.concatenate(output_row_list, 0)
#     target_img = np.concatenate(target_row_list, 0)
#     concat_img = np.concatenate(concat_row_list, 0)
#     cv2.imwrite(os.path.join(save_dir, '%07d_output.png' % (i + 1)), output_img)
#     cv2.imwrite(os.path.join(save_dir, '%07d_target.png' % (i + 1)), target_img)
#     cv2.imwrite(os.path.join(save_dir, '%07d.png' % (i + 1)), concat_img)


# data_dir = 'results/243/contrast_agis_80/outputs/D3'
# font_ids = [181, 50, 129, 201, 149, 106, 35, 187, 72, 77]
# char_ids = [595, 487, 747, 118, 723, 310, 796, 1106, 623, 574]
# col_list = []
# for i in range(len(font_ids)):
#     img = cv2.imread(os.path.join(data_dir, '%03d_%04d.png' % (i, char_ids[i] - 1)))
#     col_list.append(img)
# cv2.imwrite('agis_D3.png', np.concatenate(col_list, 1))


# random select 100 images of 1125 images to compute fid

# import random

# src_dir = 'results/243/contrast_agis'
# dst_dir = 'results/243/contrast_agis_100of1125'

# L = os.listdir(src_dir + '/outputs/D3')
# L.sort()

# for i in tqdm(range(61)):
#     temp_L = list(range(0, 1125))
#     sample_L = random.sample(temp_L, 100)
#     for j in range(len(sample_L)):
#         src_path = os.path.join(src_dir, 'outputs/D3', L[i*1125+sample_L[j]])
#         dst_path = os.path.join(dst_dir, 'outputs/D3', L[i*1125+sample_L[j]])
#         shutil.copy(src_path, dst_path)
#         src_path = os.path.join(src_dir, 'targets/D3', L[i*1125+sample_L[j]])
#         dst_path = os.path.join(dst_dir, 'targets/D3', L[i*1125+sample_L[j]])
#         shutil.copy(src_path, dst_path)


#################################################################################
# concat data

# data_dir = 'results/243/test_style_num3_newdis_w420_1125'
# path_L = ['encoder_1', 'encoder_2', 'encoder_3']

# for i in range(len(path_L)):
#     for dn in [1, 2, 4]:
#         save_dir = os.path.join(data_dir, path_L[i], 'D%d' % dn)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         L = os.listdir(data_dir + '/D%d' % dn)
#         for j in tqdm(range(len(L))):
#             img = cv2.imread(os.path.join(data_dir, 'D%d' % dn, L[j]))
#             subimg = img[i*400 : (i+1)*400]
#             cv2.imwrite(os.path.join(save_dir, L[j]), subimg)

# indices = [181, 50, 129, 201, 149, 106, 35, 187, 72, 77, 15, 19, 30, 59, 80, 67, 68]
# data_dir = 'datasets/243/images'
# save_dir = 'results/243/contrast_zi2zi_80/targets/D1'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# char_list = utils.load_list('datasets/243/char_list.txt')

# for i in tqdm(range(len(indices))):
#     for j in range(1125):
#         src_path = os.path.join(data_dir, '%03d' % indices[i], char_list[j])
#         dst_path = os.path.join(save_dir, '%03d_%04d.png' % (i, j))
#         shutil.copy(src_path, dst_path)


# data_dir = 'results/243/contrast_zi2zi_256/outputs/D1'
# save_dir = 'results/243/contrast_zi2zi_80/outputs/D1'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# L = os.listdir(data_dir)
# for i in tqdm(range(len(L))):
#     img = cv2.imread(os.path.join(data_dir, L[i]))
#     img = cv2.resize(img, (80, 80))
#     cv2.imwrite(os.path.join(save_dir, L[i]), img)



# font_ids = [139, 39, 98, 154, 113, 83, 27, 144, 58, 62]
# char_ids = [595, 487, 747, 118, 723, 310, 796, 1106, 623, 574]
# data_dir = 'results/243/test_1125/targets/D1'
# img_list = []
# for i in range(len(font_ids)):
#     first_id = (font_ids[i] * 1125 + char_ids[i]) // 50
#     second_id = (font_ids[i] * 1125 + char_ids[i]) % 50
#     path = os.path.join(data_dir, '%07d_target_%02d.png' % (first_id + 1, second_id - 1))
#     img = cv2.imread(path)
#     img_list.append(img)
# cv2.imwrite('target_D1.png', np.concatenate(img_list, 1))


# img_ids = [23, 23, 23, 69, 69, 217, 220, 273, 282, 282, 320, 333, 333, 590, 590, 977]
# char_ids = [33, 36, 44, 9, 17, 18, 12, 23, 15, 47, 32, 8, 17, 19, 26, 12]
# data_dir = 'results/243/contrast_agis_80/D3'
# img_list = []
# for i in range(len(char_ids)):
#     path = os.path.join(data_dir, '%07d_output.png' % img_ids[i])
#     img = cv2.imread(path)
#     row, col = (char_ids[i] - 1) // 10, (char_ids[i] - 1) % 10
#     img_list.append(img[80*row : 80*(row+1), 80*col : 80*(col+1)])
# cv2.imwrite('agis_D3.png', np.concatenate(img_list, 1))

#################################################################################
# contrast lffont

# data_path = 'results/243/contrast_lffont'
# L = os.listdir(data_path)

# for p in L:
#     path = os.path.join(data_path, p)
#     sub_L = os.listdir(path)
#     for img_name in sub_L:
#         img_path = os.path.join(path, img_name)
#         if "target" in img_name:
#             img = cv2.imread(img_path)
#             cv2.imwrite(os.path.join(path, img_name[:7] + '.png'), img)

################################################################################
# convert zi2zi interp data to 80 resol, add target image

# data_path = 'results/243/contrast_zi2zi_80/interp256'
# output_path = 'results/243/contrast_zi2zi_80/interp80'
# resol = 80
# L = os.listdir(data_path)
# font_list = [181, 50, 129, 201, 149, 106, 35, 187, 72, 77, 15, 19, 30, 59, 80, 67, 68]
# char_list = utils.load_list('datasets/243/char_list.txt')
# for i in range(len(L)):
#         font1, font2 = map(int, L[i].split('to'))
#         img_list = os.listdir(os.path.join(data_path, L[i]))
#         if not os.path.exists(os.path.join(output_path, L[i])):
#                 os.makedirs(os.path.join(output_path, L[i]))
#         for j in range(len(img_list)):
#                 img_path = os.path.join(data_path, L[i], img_list[j])
#                 img = cv2.imread(img_path)
#                 img = cv2.resize(img, (resol*11, resol))
#                 left_img = cv2.imread(os.path.join('datasets/243/images/%03d/%s' % (font_list[font1], char_list[j])))
#                 right_img = cv2.imread(os.path.join('datasets/243/images/%03d/%s' % (font_list[font2], char_list[j])))
#                 concat_img = np.concatenate([left_img, img, right_img], 1)
#                 cv2.imwrite(os.path.join(output_path, L[i], img_list[j]), concat_img)

################################################################################
# cut certain chars according to ids

# def cut_and_save(save_dir, L, num_per_char, resol=80):
#         cnt = 0
#         for i in range(len(L)):
#                 img_path = L[i][0]
#                 img = cv2.imread(img_path, 0)
#                 for j in range(1, len(L[i])):
#                         row, col = L[i][j]
#                         sub_img = img[(row-1)*resol:row*resol, (col-1)*num_per_char*resol:col*num_per_char*resol]
#                         rows = []
#                         for k in range(num_per_char):
#                                 rows.append(sub_img[:, k*resol:(k+1)*resol])
#                         concat_img = np.concatenate(rows, 0)
#                         cv2.imwrite(os.path.join(save_dir, '%04d.png' % cnt), concat_img)
#                         cnt += 1

# save_dirs = ['results/cut_results/multistyle_D1', 'results/cut_results/multistyle_D3',
#         'results/cut_results/dis_D1', 'results/cut_results/dis_D3',
#         'results/cut_results/ms+dis_D1', 'results/cut_results/ms+dis_D3',
#         'results/cut_results/contrast_D2', 'results/cut_results/contrast_D3']
# nums = [5, 5, 7, 7, 5, 5, 4, 4]
# total_L = [
#         [
#                 ['results/concat_multistyle/D1/0000002.png', [5, 2], [12, 2], [15,1]],
#                 ['results/concat_multistyle/D1/0000005.png', [18,2]],
#                 ['results/concat_multistyle/D1/0000007.png', [14,1]],
#                 ['results/concat_multistyle/D1/0000011.png', [10,1], [18,2]],
#                 ['results/concat_multistyle/D1/0000013.png', [14,2]],
#                 ['results/concat_multistyle/D1/0000035.png', [19,2]],
#                 ['results/concat_multistyle/D1/0000051.png', [14,2]],
#         ],
#         [
#                 ['results/concat_multistyle/D3/0000002.png', [23,2]],
#                 ['results/concat_multistyle/D3/0000003.png', [24,1]],
#                 ['results/concat_multistyle/D3/0000005.png', [4,2]],
#                 ['results/concat_multistyle/D3/0000009.png', [12,2]],
#                 ['results/concat_multistyle/D3/0000011.png', [12,1]],
#                 ['results/concat_multistyle/D3/0000014.png', [12,2], [23,2]],
#                 ['results/concat_multistyle/D3/0000020.png', [25,2]],
#                 ['results/concat_multistyle/D3/0000012.png', [8,2]],
#                 ['results/concat_multistyle/D3/0000013.png', [6,2]],
#         ],
#         [
#                 ['results/concat_newdis/D1/0000001.png', [1,1],[13,1]],
#                 ['results/concat_newdis/D1/0000002.png', [6,1],[15,1],[23,2]],
#                 ['results/concat_newdis/D1/0000004.png', [19,1]],
#                 ['results/concat_newdis/D1/0000005.png', [13,2]],
#                 ['results/concat_newdis/D1/0000012.png', [8,1]],
#                 ['results/concat_newdis/D1/0000019.png', [8,1]],
#                 ['results/concat_newdis/D1/0000008.png', [25,2]],
#         ],
#         [
#                 ['results/concat_newdis/D3/0000001.png', [5,2],[22,2]],
#                 ['results/concat_newdis/D3/0000002.png', [14,1],[17,2]],
#                 ['results/concat_newdis/D3/0000003.png', [9,2],[11,2]],
#                 ['results/concat_newdis/D3/0000005.png', [4,1],[19,1]],
#                 ['results/concat_newdis/D3/0000007.png', [9,2]],
#                 ['results/concat_newdis/D3/0000010.png', [9,1]],
#         ],
#         [
#                 ['results/concat_ms+dis/D1/0000001.png', [11,1]],
#                 ['results/concat_ms+dis/D1/0000003.png', [3,1],[8,2]],
#                 ['results/concat_ms+dis/D1/0000004.png', [7,1],[17,2]],
#                 ['results/concat_ms+dis/D1/0000005.png', [22,2]],
#                 ['results/concat_ms+dis/D1/0000010.png', [22,1]],
#                 ['results/concat_ms+dis/D1/0000011.png', [10,1]],
#                 ['results/concat_ms+dis/D1/0000014.png', [11,1],[5,2]],
#         ],
#         [
#                 ['results/concat_ms+dis/D3/0000010.png', [1,1],[20,2]],
#                 ['results/concat_ms+dis/D3/0000012.png', [12,2]],
#                 ['results/concat_ms+dis/D3/0000014.png', [13,1]],
#                 ['results/concat_ms+dis/D3/0000020.png', [4,2]],
#                 ['results/concat_ms+dis/D3/0000021.png', [14,2]],
#                 ['results/concat_ms+dis/D3/0000023.png', [11,1],[18,1]],
#                 ['results/concat_ms+dis/D3/0000024.png', [25,2]],
#                 ['results/concat_ms+dis/D3/0000030.png', [19,1]],
#         ],
#         [
#                 ['results/concat_contrast_D123/D2/0000009.png', [8,1]],
#                 ['results/concat_contrast_D123/D2/0000010.png', [1,1],[7,1],[20,2]],
#                 ['results/concat_contrast_D123/D2/0000012.png', [14,1]],
#                 ['results/concat_contrast_D123/D2/0000047.png', [4,1],[4,2]],
#                 ['results/concat_contrast_D123/D2/0000048.png', [22,1]],
#                 ['results/concat_contrast_D123/D2/0000049.png', [18,1]],
#                 ['results/concat_contrast_D123/D2/0000052.png', [24,1]],
#                 ['results/concat_contrast_D123/D2/0000086.png', [5,1]],
#                 ['results/concat_contrast_D123/D2/0000087.png', [25,2]],
#                 ['results/concat_contrast_D123/D2/0000088.png', [15,2]],
#                 ['results/concat_contrast_D123/D2/0000089.png', [15,1]],
#                 ['results/concat_contrast_D123/D2/0000090.png', [17,2]],
#                 ['results/concat_contrast_D123/D2/0000128.png', [15,2]],
#                 ['results/concat_contrast_D123/D2/0000129.png', [9,1]],
#                 ['results/concat_contrast_D123/D2/0000132.png', [12,1],[22,2]],
#                 ['results/concat_contrast_D123/D2/0000033.png', [5,1]],
#                 ['results/concat_contrast_D123/D2/0000057.png', [17,2]]
#         ],
#         [
#                 ['results/concat_contrast_D123/D3/0000115.png', [12,1]],
#                 ['results/concat_contrast_D123/D3/0000125.png', [9,2]],
#                 ['results/concat_contrast_D123/D3/0000579.png', [4,1]],
#                 ['results/concat_contrast_D123/D3/0000584.png', [12,1]],
#         ]
# ]
# for i in trange(len(save_dirs)):
#         if not os.path.exists(save_dirs[i]):
#                 os.makedirs(save_dirs[i])
#         cut_and_save(save_dirs[i], total_L[i], nums[i])

################################################################################

# base_dir = 'results/243'
# data_dirs = ['interp_HKLS_80/HKL/D1', 'interp_HKLS_80/OLF/D1', 'interp_HKLS_80/SHK/D1', 
#         'interp_HKLS_80/SHK/D1', 'interp_3style_tencent_num100/avg_interp_12']
# base_rows = [0, 0, 0, 5, 0]
# save_dir_names = ['HK', 'OL', 'SH', 'SK', 'tencent']

# for i in trange(len(data_dirs)):
#         save_dir = os.path.join('results/243/contrast_msemd_interp', save_dir_names[i])
#         if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#         load_dir = os.path.join(base_dir, data_dirs[i])
#         L = os.listdir(load_dir)
#         L.sort()
#         cnt = 0
#         for j in range(len(L)):
#                 img = cv2.imread(os.path.join(load_dir, L[j]), 0)
#                 for k in range(base_rows[i], base_rows[i]+5):
#                         sub_img = img[k*80:(k+1)*80]
#                         cv2.imwrite(os.path.join(save_dir, '%04d.png' % cnt), sub_img)
#                         cnt += 1

#################################################################################

data_dirs = ['results/243/contrast_zi2zi_80/interp', 'results/243/contrast_agis_80/interp',
        'results/243/contrast_emd_interp', 'results/243/contrast_msemd_interp']
dir_names = ['HK', 'OL', 'SH', 'SK', 'tencent']

for i in trange(len(dir_names)):
        save_dir = os.path.join('results/contrast_interp', dir_names[i])
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for j in range(1125):
                rows = []
                for k in range(len(data_dirs)):
                        img = cv2.imread(os.path.join(data_dirs[k], dir_names[i], '%04d.png' % j), 0)
                        rows.append(img)
                concat_img = np.concatenate(rows, 0)
                cv2.imwrite(os.path.join(save_dir, '%04d.png' % j), concat_img)