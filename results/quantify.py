import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import fid

def compute_iou(img1, img2):
    img1_change = np.where(img1 < 127, img1, -1)
    img2_change = np.where(img2 < 127, img2, -2)
    black_num1 = img1[img1 < 127].shape[0]
    black_num2 = img2[img2 < 127].shape[0]
    intersection_num = img1_change[img1_change == img2_change].shape[0]
    total_num = black_num1 + black_num2 - intersection_num
    # print('1:', total_num, intersection_num, black_num1, black_num2)

    return intersection_num / total_num

def compute_iou2(img1, img2):
    intersection_num = 0
    total_num = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i, j] < 127 and img2[i, j] < 127:
                total_num += 1
                if img1[i, j] == img2[i, j]:
                    intersection_num += 1
            elif img1[i, j] < 127 or img2[i, j] < 127:
                total_num += 1
    # print('2:', total_num, intersection_num)
    return intersection_num / total_num

def compute_rmse(img1, img2):
    img1 = img1 / 255
    img2 = img2 / 255
    mse = np.mean((img1 - img2) ** 2)
    return np.sqrt(mse)

def compute_l1_loss(img1, img2):
    img1 = img1 / 255
    img2 = img2 / 255
    return np.mean(np.abs(img1 - img2))

base_dir = '243'
# path_L = ['test', 'test_dis_w10', 'test_dis_w100', 'test_dis_w200',
#     'test_dis_w400', 'test_style_num2', 'test_style_num3', 'test_style_num4']
# path_L = ['test_newdis_w10', 'test_newdis_w50', 'test_newdis_w100', 'test_newdis_w200',
#     'test_newdis_w400', 'test_newdis_w600', 'test_newdis_w800']
path_L = ['contrast_zi2zi_80']
save_dir = 'quantification'
save_name = 'quantify_zi2zi_80'
# quantification indicators
q_indicators = ['iou', 'rmse', 'l1_loss', 'ssim', 'psnr']
# q_indicators = ['rmse', 'ssim', 'psnr']
resol = 80
AREA = [1, 2, 3, 4]
f = open(save_dir + '/%s.csv' % save_name, 'w')

# dn: area in [1, 2, 3, 4]
def quantify_area(dn, plot=True):
    f.write('D%d\n' % dn)
    f.write(',')
    # quantification dict
    q_dict = {}
    for indicator in q_indicators:
        q_dict[indicator] = []
    for i in range(len(path_L)):
        f.write(path_L[i])
        if i != len(path_L) - 1:
            f.write(',')
        else:
            f.write('\n')
        for indicator in q_indicators:
            q_dict[indicator].append([])

    img_list = os.listdir(os.path.join(base_dir, path_L[0], 'D%d' % dn))
    img_list.sort()
    for i in tqdm(range(len(img_list))):
        if i % 3 == 0:
            for j in range(len(path_L)):
                img_path = os.path.join(base_dir, path_L[j], 'D%d' % dn, img_list[i+1])
                output_img = cv2.imread(img_path, 0)
                target_path = os.path.join(base_dir, path_L[j], 'D%d' % dn, img_list[i+2])
                target_img = cv2.imread(target_path, 0)
                iou = compute_iou(output_img, target_img)
                rmse = compute_rmse(output_img, target_img)
                l1_loss = compute_l1_loss(output_img, target_img)
                ssim = structural_similarity(output_img, target_img)
                psnr = peak_signal_noise_ratio(target_img, output_img)
                q_dict['iou'][j].append(iou)
                q_dict['rmse'][j].append(rmse)
                q_dict['l1_loss'][j].append(l1_loss)
                q_dict['ssim'][j].append(ssim)
                q_dict['psnr'][j].append(psnr)
    # write to file
    for i in range(len(q_indicators)):
        indicator = q_indicators[i]
        f.write(indicator + ',')
        for j in range(len(path_L)):
            mean = np.mean(q_dict[indicator][j])
            f.write(str(mean))
            if j != len(path_L) - 1:
                f.write(',')
            else:
                f.write('\n')
    f.write('\n')
    # plot figures
    if plot:
        plt.figure(figsize=(12, 12), dpi=256)
        plt.figure(dn)
        x = np.linspace(1, 200, 200)
        for i in range(len(q_indicators)):
            indicator = q_indicators[i]
            # plot figure of indicator
            ax = plt.subplot(2, 3, i + 1)
            for j in range(len(path_L)):
                ax.plot(x, q_dict[indicator][j], label=path_L[j])
            ax.set_xlabel('x label')
            ax.set_ylabel(indicator)
            ax.set_title('%ss' % indicator.upper())
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.savefig(save_dir + '/%s_D%d.png' % (save_name, dn))
    


def compute_quantification(output_img, target_img, indicator):
    if indicator == 'iou':
        return compute_iou(output_img, target_img)
    elif indicator == 'rmse':
        return compute_rmse(output_img, target_img)
    elif indicator == 'l1_loss':
        return compute_l1_loss(output_img, target_img)
    elif indicator == 'ssim':
        return structural_similarity(output_img, target_img)
    elif indicator == 'psnr':
        return peak_signal_noise_ratio(target_img, output_img)
    else:
        raise Exception('Wrong indicator!')

def quantify_by_indicator(idx, plot=True):
    indicator = q_indicators[idx]
    f.write(indicator + '\n')
    f.write(',')
    # quantification list
    q_list = []
    for i in range(4):
        q_list.append([])
    for i in range(len(path_L)):
        f.write(path_L[i])
        if i != len(path_L) - 1:
            f.write(',')
        else:
            f.write('\n')
        for j in range(4):
            q_list[j].append([])

    for dn in AREA:
        img_list = os.listdir(os.path.join(base_dir, path_L[0], 'D%d' % dn))
        img_list.sort()
        for i in tqdm(range(len(img_list))):
            if i % 3 == 0:
                for j in range(len(path_L)):
                    img_path = os.path.join(base_dir, path_L[j], 'D%d' % dn, img_list[i+1])
                    output_img = cv2.imread(img_path, 0)
                    target_path = os.path.join(base_dir, path_L[j], 'D%d' % dn, img_list[i+2])
                    target_img = cv2.imread(target_path, 0)
                    ans = compute_quantification(output_img, target_img, indicator)
                    q_list[dn-1][j].append(ans)
        f.write('D%d,' % dn)
        for i in range(len(path_L)):
            mean = np.mean(q_list[dn-1][i])
            f.write(str(mean))
            if i != len(path_L) - 1:
                f.write(',')
            else:
                f.write('\n')
    f.write('\n')
    # plot figures
    if plot:
        plt.figure(figsize=(12, 12), dpi=256)
        plt.figure(idx + 1)
        x = np.linspace(1, 200, 200)
        for dn in range(1, 5):
            ax = plt.subplot(2, 3, dn)
            for i in range(len(path_L)):
                ax.plot(x, q_list[dn-1][i], label=path_L[i])
            ax.set_xlabel('x label')
            ax.set_ylabel(indicator)
            ax.set_title('%s of D%d' % (indicator.upper(), dn))
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.savefig(save_dir + '/%s_%s.png' % (save_name, indicator))
        f.write('\n')

############################## calculate fid ###################################
def separate_and_save(src_path, dst_path, label='output', row=5, col=10, resol=128):
    if os.path.exists(dst_path):
        return
    os.makedirs(dst_path)
    for name in os.listdir(src_path):
        if label in name:
            img = cv2.imread(src_path + '/' + name, 0)
            for i in range(row):
                for j in range(col):
                    sub_img = img[i*resol : (i+1)*resol, j*resol : (j+1)*resol]
                    cv2.imwrite(dst_path + '/%s_%02d.png' % (name[:-4], i*col+j), sub_img)

def calculate_fid():
    f.write('fid\n')
    f.write(',')
    for i in range(len(path_L)):
        f.write(path_L[i])
        if i != len(path_L) - 1:
            f.write(',')
        else:
            f.write('\n')
    for dn in AREA:
        print('====================processing D%d...=====================' % dn)
        f.write('D%d,' % dn)
        for i in tqdm(range(len(path_L))):
            output_dir = os.path.join(base_dir, path_L[i], 'outputs/D%d' % dn)
            target_dir = os.path.join(base_dir, path_L[i], 'targets/D%d' % dn)
            #if not os.path.exists(output_dir):
            src_path = os.path.join(base_dir, path_L[i], 'D%d' % dn)
            separate_and_save(src_path, output_dir, 'output', resol=resol)
            separate_and_save(src_path, target_dir, 'target', resol=resol)
            fid_value = fid.calculate_fid_given_paths([output_dir, target_dir], None)
            f.write(str(fid_value))
            if i != len(path_L) - 1:
                f.write(',')
            else:
                f.write('\n')

###############################################################################

for idx in range(len(q_indicators)):
    print('computing %s...' % q_indicators[idx])
    quantify_by_indicator(idx, plot=False)
calculate_fid()

f.close()