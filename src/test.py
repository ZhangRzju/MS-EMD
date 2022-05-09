import pickle
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tqdm import tqdm
import json

import networks
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, 
    help='directory to load image path data')
parser.add_argument('--output_dir', type=str, 
    help='directory to save model and training results')
parser.add_argument('--checkpoint', type=str, default=None,
    help='directory to resume training')
parser.add_argument('--input_size', type=int, default=80, choices=[80, 256],
    help='size of input images for emd model')
parser.add_argument('--batch_size', type=int, default=50,
    help='number of target images in one batch')
parser.add_argument('--save_size', type=int, nargs='+', default=[5, 10], 
    help='number of rows and columns of output images')

parser.add_argument('--with_skel', type=int, default=0, choices=[0, 1, 2, 3],
    help='the way to use skeleton data, 0 means skeleton data is not used')
parser.add_argument('--style_num', type=int, default=1, choices=[2, 3, 4],
    help='the number of style encoder in emd networks')
parser.add_argument('--decoder_base_channel', type=int, default=64,
    help='base channel to be multiplied in decoder')
parser.add_argument('--decoder_num', type=int, default=1, choices=[2, 3, 4],
    help='number of decoder in multi-style training')

parser.add_argument('--with_dis', type=int, default=0, choices=[0, 1],
    help='add discriminator or not')

parser.add_argument('--with_local_enhancer', type=int, default=0, choices=[0, 1],
    help='use local enhancer net or not')
parser.add_argument('--local_enhancer_with_dis', type=int, default=0, choices=[0, 1],
    help='add discriminator behind local enhancer net or not')

parser.add_argument('--interp', type=int, default=0, choices=[0, 1],
    help='test with interpolation or not')
parser.add_argument('--interp_interval', type=float, nargs='+', default=[0, 1],
    help='interval of interpolation')
parser.add_argument('--interp_num', type=int, default=10,
    help='number of interpolation images between several styles')
parser.add_argument('--interp_3style', type=int, default=0,
    help='interpolate 3 styles simultaneously or not')
args = parser.parse_args()


    
def test(dn):
    input_dir = args.input_dir + '/D%d' % dn

    with open(input_dir + '/style.pkl', 'rb') as f:
        path_s = pickle.load(f)
    with open(input_dir + '/content.pkl', 'rb') as f:
        path_c = pickle.load(f)
    with open(input_dir + '/target.pkl', 'rb') as f:
        path_t = pickle.load(f)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset_s = tf.data.Dataset.from_tensor_slices(path_s) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .batch(10) \
                    .map(utils.concat_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(args.batch_size*args.style_num)
    iter_s = iter(dataset_s)

    dataset_c = tf.data.Dataset.from_tensor_slices(path_c) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .batch(10) \
                    .map(utils.concat_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(args.batch_size)
    iter_c = iter(dataset_c)

    dataset_t = tf.data.Dataset.from_tensor_slices(path_t) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(args.batch_size*args.style_num)
    iter_t = iter(dataset_t)

    model = networks.Net(args)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.checkpoint))

    img_save_dir = os.path.join(args.output_dir, 'D%d' % dn)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    max_steps = int(len(path_t) / (args.batch_size * args.style_num))

    print('generating images in D%d area...' % dn)
    for step in tqdm(range(max_steps)):
        batch_s = iter_s.get_next()
        batch_c = iter_c.get_next()
        batch_t = iter_t.get_next()

        if args.interp:
            if args.interp_3style:
                output_imgs = model.generator.interpolate_3style([batch_s, batch_c], args)
                utils.save_interp_images(output_imgs, batch_t, args, step, img_save_dir, add_target=False)
            else:
                output_imgs = model.generator.interpolate([batch_s, batch_c], args)
                utils.save_interp_images(output_imgs, batch_t, args, step, img_save_dir)
        else:
            output_imgs = model.generator([batch_s, batch_c], training=False)
            utils.save_results(output_imgs, batch_t, args, step, img_save_dir, 1)


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    utils.clear_dir(args.output_dir)

with open(os.path.join(args.output_dir, "options.json"), "w") as f:
    arg_dict = vars(args)
    arg_dict['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    f.write(json.dumps(arg_dict, sort_keys=True, indent=4))


test(1)
if not args.interp:
    test(2)
    test(3)
    test(4)