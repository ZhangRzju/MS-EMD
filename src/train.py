import pickle
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tqdm import tqdm
import random
import numpy as np
import json
import traceback

import networks
import utils
import logger


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, 
    help='directory to load image path data')
parser.add_argument('--output_dir', type=str, 
    help='directory to save model and training results')
parser.add_argument('--seed', type=int,
    help='seed to initialize a random function')
parser.add_argument('--save_summary', type=int, default=0, choices=[0, 1],
    help='save summary to logdir or not')
parser.add_argument('--checkpoint', type=str, default=None,
    help='directory to resume training')
parser.add_argument('--epochs', type=int, default=200, 
    help='number of training epochs')
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
parser.add_argument('--L1_weight', type=int, default=420,
    help='weight for emd loss in generator loss')

parser.add_argument('--with_local_enhancer', type=int, default=0, choices=[0, 1],
    help='use local enhancer net or not')
parser.add_argument('--global_checkpoint', type=str, default=None,
    help='directory to load global networks')
parser.add_argument('--together_epoch', type=int, default=0,
    help='train the entire network after this epoch')
parser.add_argument('--local_enhancer_with_dis', type=int, default=0, choices=[0, 1],
    help='add discriminator behind local enhancer net or not')
parser.add_argument('--local_enhancer_L1weight', type=int, default=320,
    help='weight for L1 loss in local enhancer loss function')
args = parser.parse_args()



def train(train_logger):
    with open(args.input_dir + '/style.pkl', 'rb') as f:
        path_s = pickle.load(f)
    with open(args.input_dir + '/content.pkl', 'rb') as f:
        path_c = pickle.load(f)
    with open(args.input_dir + '/target.pkl', 'rb') as f:
        path_t = pickle.load(f)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset_s = tf.data.Dataset.from_tensor_slices(path_s) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .batch(10) \
                    .map(utils.concat_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(args.batch_size*args.style_num).repeat(args.epochs)
    iter_s = iter(dataset_s)

    dataset_c = tf.data.Dataset.from_tensor_slices(path_c) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .batch(10) \
                    .map(utils.concat_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(args.batch_size).repeat(args.epochs)
    iter_c = iter(dataset_c)

    dataset_t = tf.data.Dataset.from_tensor_slices(path_t) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(args.batch_size*args.style_num).repeat(args.epochs)
    iter_t = iter(dataset_t)


    model = networks.Net(args)
    if args.global_checkpoint:
        print('loading global model...')
        generator_ckpt = tf.train.Checkpoint(generator=model.generator.global_net)
        global_ckpt = tf.train.Checkpoint(model=generator_ckpt)
        status = global_ckpt.restore(tf.train.latest_checkpoint(args.global_checkpoint))
        # status.assert_consumed()
        print('ok.')
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, args.output_dir, max_to_keep=3)
    if args.checkpoint:
        status = ckpt.restore(tf.train.latest_checkpoint(args.checkpoint))
        # status.assert_consumed()
    
    img_save_dir = os.path.join(args.output_dir, 'images')
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if args.save_summary:
        writer = tf.summary.create_file_writer(args.summary_dir)
    # start training
    steps_per_epoch = int(len(path_t) / (args.batch_size * args.style_num))
    max_steps = args.epochs * steps_per_epoch

    for step in tqdm(range(max_steps)):
        batch_s = iter_s.get_next()
        batch_c = iter_c.get_next()
        batch_t = iter_t.get_next()
        
        epoch_now = step // steps_per_epoch + 1
        if epoch_now > args.together_epoch:
            output_imgs, loss_dict = model.train_step(batch_s, batch_c, batch_t) 
        else:
            output_imgs, loss_dict = model.train_step_local(batch_s, batch_c, batch_t)

        if (step + 1) % 50 == 0:
            if args.save_summary:
                utils.add_summary(writer, loss_dict, step)
            utils.print_losses(args, train_logger, loss_dict, step, steps_per_epoch)
        
        if (step + 1) % 500 == 0:
            print('saving images...')
            utils.save_results(output_imgs, batch_t, args, step, img_save_dir)

        if (step + 1) % steps_per_epoch == 0:
            print('saving checkpoint...')
            manager.save()


def train_multi_gpu(train_logger):
    strategy = tf.distribute.MirroredStrategy()
    train_logger.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync

    with open(args.input_dir + '/style.pkl', 'rb') as f:
        path_s = pickle.load(f)
    with open(args.input_dir + '/content.pkl', 'rb') as f:
        path_c = pickle.load(f)
    with open(args.input_dir + '/target.pkl', 'rb') as f:
        path_t = pickle.load(f)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset_s = tf.data.Dataset.from_tensor_slices(path_s) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .batch(10) \
                    .map(utils.concat_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(global_batch_size*args.style_num).repeat(args.epochs)
    dist_dataset_s = strategy.experimental_distribute_dataset(dataset_s)
    iter_s = iter(dist_dataset_s)

    dataset_c = tf.data.Dataset.from_tensor_slices(path_c) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .batch(10) \
                    .map(utils.concat_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(global_batch_size).repeat(args.epochs)
    dist_dataset_c = strategy.experimental_distribute_dataset(dataset_c)
    iter_c = iter(dist_dataset_c)

    dataset_t = tf.data.Dataset.from_tensor_slices(path_t) \
                    .map(utils.load_images, num_parallel_calls=AUTOTUNE) \
                    .prefetch(AUTOTUNE) \
                    .batch(global_batch_size*args.style_num).repeat(args.epochs)
    dist_dataset_t = strategy.experimental_distribute_dataset(dataset_t)
    iter_t = iter(dist_dataset_t)


    with strategy.scope():
        model = networks.Net(args, strategy.num_replicas_in_sync)
        if args.global_checkpoint:
            print('loading global model...')
            generator_ckpt = tf.train.Checkpoint(generator=model.generator.global_net)
            global_ckpt = tf.train.Checkpoint(model=generator_ckpt)
            status = global_ckpt.restore(tf.train.latest_checkpoint(args.global_checkpoint))
            # status.assert_consumed()
            print('ok.')
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, args.output_dir, max_to_keep=3)
        if args.checkpoint:
            status = ckpt.restore(tf.train.latest_checkpoint(args.checkpoint))
            # status.assert_consumed()
    
    img_save_dir = os.path.join(args.output_dir, 'images')
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if args.save_summary:
        writer = tf.summary.create_file_writer(args.summary_dir)
    # start training
    steps_per_epoch = int(len(path_t) / (global_batch_size * args.style_num))
    max_steps = args.epochs * steps_per_epoch

    for step in tqdm(range(max_steps)):
        batch_s = iter_s.get_next()
        batch_c = iter_c.get_next()
        batch_t = iter_t.get_next()
        
        epoch_now = step // steps_per_epoch + 1
        if epoch_now > args.together_epoch:
            output_imgs, loss_dict = strategy.run(model.train_step, 
                args=(batch_s, batch_c, batch_t))
        else:
            output_imgs, loss_dict = strategy.run(model.train_step_local, 
                args=(batch_s, batch_c, batch_t))

        if (step + 1) % 50 == 0:
            for key, value in loss_dict.items():
                loss_dict[key] = strategy.reduce(tf.distribute.ReduceOp.SUM, value, axis=None) 
            if args.save_summary:
                utils.add_summary(writer, loss_dict, step)
            utils.print_losses(args, train_logger, loss_dict, step, steps_per_epoch)
        
        if (step + 1) % 500 == 0:
            print('saving images...')
            utils.save_results(output_imgs, batch_t, args, step, img_save_dir, 
                strategy.num_replicas_in_sync)

        if (step + 1) % steps_per_epoch == 0:
            print('saving checkpoint...')
            manager.save()



def main():
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # clean or create output directory 
    if os.path.exists(args.output_dir):
        utils.clear_dir(args.output_dir)
    else:
        os.makedirs(args.output_dir)
    # save options
    arg_dict = vars(args)
    arg_dict['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(arg_dict, sort_keys=True, indent=4))
    
    # directory to save summary
    args.summary_dir = args.output_dir.replace('results', 'runs')
    if args.save_summary:
        if os.path.exists(args.summary_dir):
            utils.clear_dir(args.summary_dir, remind=False)
        else:
            os.makedirs(args.summary_dir)
        with open(os.path.join(args.summary_dir, "options.json"), "w") as f:
            f.write(json.dumps(arg_dict, sort_keys=True, indent=4))
    
    train_logger = logger.create_logger(os.path.join(args.output_dir, 'train.log'))
    train_logger.info('============ Initialized logger ============')
    train_logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v 
                            in sorted(vars(args).items()) ) )

    num_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    try:
        # if num_gpu == 1:
        #     train(train_logger)
        # else:
        #     train_multi_gpu(train_logger)
        train_multi_gpu(train_logger)
    except Exception as e:
        train_logger.error(e)
        train_logger.error(traceback.format_exc())


main()