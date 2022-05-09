import cv2
import numpy as np
import tensorflow as tf
import os
import shutil

# 读取随机打乱的font和char列表文件
def load_list(filepath):
    L = []
    with open(filepath, encoding='utf8') as f:
        for line in f:
            L.append(line.strip())
    return L

# 清空dir_path
def clear_dir(dir_path, remind=True):
    if remind:
        _ = input('clear %s or not?' % dir_path)
    print('clearing ' + dir_path + ' ...')
    for f in os.listdir(dir_path):
        filepath = dir_path + '/' + f
        if os.path.isfile(filepath):
            os.remove(filepath)
        else:
            shutil.rmtree(filepath, True)
    print('ok.')

#用于tf.data.Dataset.map
def load_images(inputs):
    image_string = tf.io.read_file(inputs)
    try:
        images = tf.image.decode_jpeg(image_string, channels=1)
    except:
        images = tf.image.decode_png(image_string, channels=1)
    images = tf.cast(images, tf.float32)
    return images / 255.0

#用于tf.data.Dataset.map
def concat_images(inputs):
    images = []
    for i in range(10):
        images.append(inputs[i])
    
    return tf.concat(images, -1)

def rearrange_images(images, batch_size, n):
    b, h, w, c = images.shape
    assert b == batch_size * n
    
    outputs = []
    for i in range(n):
        for j in range(batch_size):
            img = images[j * n + i][None]
            outputs.append(img)
    
    return np.concatenate(outputs, 0)

def save_separate_images(images, n, size, save_path, name):
    row, col = size
    # b, h, w, _ = images.shape
    row_list = []
    for i in range(row):
        col_list = []
        for j in range(col):
            col_list.append(images[i * col + j, :, :, 0])
        col_img = np.concatenate(col_list, 1)
        row_list.append(col_img)
    row_img = np.concatenate(row_list, 0)
    cv2.imwrite(save_path + '/%07d_%s.png' % (n, name), 255 * row_img)

def save_concat_images(outputs, targets, n, size, save_path):
    row, col = size
    # b, h, w, _ = outputs.shape
    row_list = []
    for i in range(row):
        col_list = []
        for j in range(col):
            col_list.append(targets[i * col + j, :, :, 0])
            col_list.append(outputs[i * col + j, :, :, 0])
        col_img = np.concatenate(col_list, 1)
        row_list.append(col_img)
    row_img = np.concatenate(row_list, 0)
    cv2.imwrite(save_path + '/%07d.png' % n, 255 * row_img)


def save_results(outputs, targets, args, step, save_path, num_gpu=1):
    row, col = args.save_size
    assert row * col == args.batch_size
    save_size = [row * args.style_num * num_gpu, col]
    if num_gpu > 1:
        outputs = tf.concat(outputs.values, 0)
        target_list = []
        for i in range(num_gpu):
            target_imgs = rearrange_images(targets.values[i], args.batch_size, args.style_num)
            target_list.append(target_imgs)
        targets = tf.concat(target_list, 0)
    else:
        targets = rearrange_images(targets, args.batch_size, args.style_num)
    save_separate_images(outputs, step + 1, save_size, save_path, 'output')
    save_separate_images(targets, step + 1, save_size, save_path, 'target')
    save_concat_images(outputs, targets, step + 1, save_size, save_path)

# save interpolation results, style number is 3.
def save_interp_images(outputs, targets, args, step, save_path, add_target=True):
    row, col = args.save_size
    assert row * col == args.batch_size
    targets = rearrange_images(targets, args.batch_size, args.style_num)
    targets_by_style = [targets[i * args.batch_size : (i+1) * args.batch_size]
        for i in range(args.style_num)]
    
    # interpolation pairs, each number represents a style
    interp_pairs = [[0, 1], [0, 2], [1, 2]]
    img_list = []
    for pair_n in range(len(interp_pairs)):
        for i in range(row):
            for j in range(col):
                s1, s2 = interp_pairs[pair_n]
                if add_target:
                    img_list.append(targets_by_style[s1][i*col+j][None])
                for k in range(args.interp_num + 1):
                    img_list.append(outputs[k][pair_n*args.batch_size+i*col+j][None])
                if add_target:
                    img_list.append(targets_by_style[s2][i*col+j][None])
    
    images = np.concatenate(img_list, 0)
    save_size = [row*args.style_num, col*(args.interp_num+1+2*add_target)]
    save_separate_images(images, step + 1, save_size, save_path, 'interp')


def add_summary(writer, loss_dict, step):
    with writer.as_default():
        for key, value in loss_dict.items():
            tf.summary.scalar(key, value, step=step+1)
        writer.flush()

def print_losses(args, train_logger, loss_dict, step, steps_per_epoch):
    epoch_now = step // steps_per_epoch + 1
    step_of_epoch = step % steps_per_epoch + 1
    loss_format = 'epoch: %03d/%03d, step: %05d/%05d, emd_loss = %f, L1_loss= %f, '\
        'mse = %f' % (epoch_now, args.epochs, step_of_epoch, steps_per_epoch, 
        loss_dict['emd_loss'], loss_dict['L1_loss'], loss_dict['mse'])
    if args.with_dis:
        loss_format += ', d_loss = %f, g_loss = %f' % (loss_dict['d_loss'], 
            loss_dict['g_loss'])
    train_logger.info(loss_format)