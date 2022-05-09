from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import numpy as np
import tensorflow as tf


def encoder_block(channel, kernel_size, stride, padding='same'):
    result = keras.Sequential([
        layers.Conv2D(channel, kernel_size, stride, padding=padding),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
    ])
    return result

def decoder_block(channel, kernel_size, stride, pad, padding='same'):
    result = keras.Sequential([
        layers.Conv2DTranspose(channel, kernel_size, stride, padding=padding,
            output_padding=pad),
        layers.BatchNormalization(), layers.ReLU(),
    ])
    return result

class Mixer(layers.Layer):
    def __init__(self, out_dim):
        super(Mixer, self).__init__()
        
        self.out_dim = out_dim
    
    def build(self, input_shape):
        w_init = tf.random_normal_initializer(0, 0.02)
        self.w = tf.Variable(
            name='w',
            initial_value=w_init(shape=(input_shape[0][-1], input_shape[1][-1], 
                self.out_dim), dtype='float32'),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name='b',
            initial_value=b_init(shape=(self.out_dim,), dtype='float32'),
            trainable=True
        )
    
    def call(self, inputs):
        input_s = inputs[0]
        input_c = inputs[1]
        outputs = tf.einsum('bi,ijk->bjk', input_s, self.w)
        outputs = tf.einsum('bj,bjk->bk', input_c, outputs)
        return outputs + self.b
        

def Encoder(input_size=80):
    channels = [64, 128, 256, 512, 512, 512, 512, 512]
    if input_size == 80:
        kernel_sizes = [5, 3, 3, 3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2, 2, 2, 2]
    elif input_size == 256:
        kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5]
        strides = [2, 2, 2, 2, 2, 2, 2, 2]
    else:
        raise Exception('illegal image size!')
    inputs = layers.Input(shape=(input_size, input_size, 10))
    outputs = [inputs]
    for i in range(8):
        x = encoder_block(channels[i], kernel_sizes[i], strides[i])(outputs[-1])
        outputs.append(x)
    return keras.Model(inputs=inputs, outputs=outputs)

class Decoder(keras.Model):
    def __init__(self, input_size=80):
        super(Decoder, self).__init__()

        self.decoderblocks=[]
        channels = [512, 512, 512, 512, 256, 128, 64, 1]
        if input_size == 80:
            kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 5]
            strides = [2, 2, 2, 2, 2, 2, 2, 1]
            pads = [1, 0, 0, 1, 1, 1, 1, 0]
        elif input_size == 256:
            kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5]
            strides = [2, 2, 2, 2, 2, 2, 2, 2]
            pads = [1, 1, 1, 1, 1, 1, 1, 1]
        else:
            raise Exception('illegal image size!')
        for i in range(7):
            self.decoderblocks.append(
                decoder_block(channels[i], kernel_sizes[i], strides[i], pads[i])
            )
        self.last_deconv = layers.Conv2DTranspose(channels[-1], kernel_sizes[-1],
            strides[-1], output_padding=pads[-1], padding='same')

    def call(self, x, c_outputs):
        outputs = [x]
        for i in range(7):
            inputs = tf.concat([outputs[-1], c_outputs[-i-1]], -1)
            x = self.decoderblocks[i](inputs)
            outputs.append(x)
        inputs = tf.concat([outputs[-1], c_outputs[-8]], -1)
        x = self.last_deconv(inputs)
        outputs.append(x)
        return outputs

class EMD(keras.Model):
    def __init__(self, input_size=80, return_penultimate=False):
        super(EMD, self).__init__()
        self.input_size = input_size
        self.return_penultimate = return_penultimate
        self.s_encoder = Encoder(input_size)
        self.c_encoder = Encoder(input_size)
        self.mix_out = 512
        self.mixer = Mixer(self.mix_out)
        self.decoder = Decoder(input_size)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        input_s = inputs[0]
        input_c = inputs[1]
        s_outputs = self.s_encoder(input_s)
        s = self.flatten(s_outputs[-1])
        c_outputs = self.c_encoder(input_c)
        c = self.flatten(c_outputs[-1])
        mix = self.mixer([s, c])
        b, h, w, _ = c_outputs[-1].get_shape()
        # use tf.reshape will raise error
        mix = layers.Reshape([h, w, self.mix_out])(mix)
        outputs = self.decoder(mix, c_outputs)
        if self.return_penultimate:
            return outputs[-2]
        else:
            return tf.sigmoid(outputs[-1])
    
    def summary(self):
        x1 = layers.Input(shape=(self.input_size, self.input_size, 10))
        x2 = layers.Input(shape=(self.input_size, self.input_size, 10))
        model = keras.Model(inputs=[x1, x2], outputs=self.call([x1, x2]))
        return model.summary()


def discriminator(input_size=80, style_condition=1, content_condition=1):
    channels = [64, 128, 256, 512, 512, 512, 512, 512]
    kernel_sizes = [5, 5, 5, 5, 5, 5, 5, 5]
    strides = [2, 2, 2, 2, 2, 2, 2, 2]
    input_s = layers.Input(shape=(input_size, input_size, 10))
    input_c = layers.Input(shape=(input_size, input_size, 10))
    inputs = layers.Input(shape=(input_size, input_size, 1))
    inputs_list = []
    if style_condition:
        inputs_list.append(input_s)
    if content_condition:
        inputs_list.append(input_c)
    inputs_list.append(inputs)
    x = tf.concat(inputs_list, -1)
    for i in range(3):
        x = encoder_block(channels[i], kernel_sizes[i], strides[i])(x)
    # output channel??
    x = layers.Conv2D(1, 3, strides=1, padding='same')(x)
    return keras.Model(inputs=inputs_list, outputs=x)

def discriminator_old(input_size=80):
    inputs = layers.Input(shape=(input_size, input_size, 1))
    x = inputs
    for i in range(3):
        x = encoder_block(64 * (i + 1), 5, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, kernel_initializer=tf.random_normal_initializer(0, 0.02))(x)
    x = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(0, 0.02))(x)

    return keras.Model(inputs=inputs, outputs=x)

class Net(keras.Model):
    def __init__(self, args, num_gpu=1):
        super(Net, self).__init__()
        self.args = args
        self.num_gpu = num_gpu
        self.change_vars = False
        self.local_vars = []
        if args.with_local_enhancer:
            if args.style_num == 1:
                self.generator = local_enhancer()
            else:
                self.generator = local_enhancer_multi_style(args.batch_size, args.style_num)
        else:
            if args.style_num == 1:
                self.generator = EMD(args.input_size)
            else:
                self.generator = EMD_multi_style(args.batch_size, args.style_num,
                    args.input_size)
        self.G_optimizer = optimizers.Adam(learning_rate=0.0002)
        if args.with_dis:
            self.style_condition = 1
            self.content_condition = 1
            self.discriminator = discriminator(args.input_size, self.style_condition,
                self.content_condition)
            # self.discriminator = discriminator_old(args.input_size)
            self.D_optimizer = optimizers.Adam(learning_rate=0.0002)
    
    @tf.function
    def train_step(self, input_s, input_c, targets):
        with tf.GradientTape(persistent=True) as tape:
            output_imgs = self.generator([input_s, input_c], training=True)
            loss_dict = self.loss_func(input_s, input_c, output_imgs, targets)
        if self.args.with_dis:
            grads = tape.gradient(loss_dict['d_loss'], self.discriminator.trainable_variables)
            self.D_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        grads = tape.gradient(loss_dict['g_loss'], self.generator.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return output_imgs, loss_dict
    
    @tf.function
    def train_step_local(self, input_s, input_c, targets):
        with tf.GradientTape(persistent=True) as tape:
            output_imgs = self.generator([input_s, input_c], training=True)
            loss_dict = self.loss_func(input_s, input_c, output_imgs, targets)
        if self.args.with_dis:
            grads = tape.gradient(loss_dict['d_loss'], self.discriminator.trainable_variables)
            self.D_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        if not self.change_vars:
            self.generator.global_net.trainable = False
            self.local_vars = self.generator.trainable_variables
            self.generator.global_net.trainable = True
            self.change_vars = True
        grads = tape.gradient(loss_dict['g_loss'], self.local_vars)
        self.G_optimizer.apply_gradients(zip(grads, self.local_vars))
        return output_imgs, loss_dict


    def loss_func(self, input_s, input_c, outputs, targets):
        targets_L = concat_data(targets, self.args.batch_size, self.args.style_num)
        targets = tf.concat(targets_L, 0)

        black = tf.greater(0.5, targets)
        black_int = tf.cast(black, tf.int32)
        black_num = tf.reduce_sum(black_int, [1,2,3]) + 1
        black_num = tf.cast(black_num, tf.float32)

        zeros = tf.zeros_like(targets)
        new_tensor = tf.where(black, targets, zeros)
        black_mean = tf.reduce_sum(new_tensor, [1,2,3]) / black_num

        weight = 1.0 / black_num * tf.nn.softmax(black_mean)
        emd_loss = tf.reduce_sum(weight*tf.reduce_sum(tf.abs(outputs - targets), [1,2,3]))
        L1_loss = tf.reduce_mean(tf.abs(outputs - targets))
        mse = tf.reduce_mean(tf.square(outputs - targets))

        loss_dict = {
            'emd_loss': emd_loss / self.num_gpu,
            'L1_loss': L1_loss / self.num_gpu,
            'mse': mse / self.num_gpu
        }

        if self.args.with_dis:
            dis_target_inputs = []
            dis_output_inputs = []
            if self.style_condition:
                input_s_list = concat_data(input_s, self.args.batch_size, self.args.style_num)
                new_input_s = tf.concat(input_s_list, 0)
                dis_target_inputs.append(new_input_s)
                dis_output_inputs.append(new_input_s)
            if self.content_condition:
                input_c_list = []
                for i in range(self.args.style_num):
                    input_c_list.append(input_c)
                new_input_c = tf.concat(input_c_list, 0)
                dis_target_inputs.append(new_input_c)
                dis_output_inputs.append(new_input_c)
            dis_target_inputs.append(targets)
            dis_output_inputs.append(outputs)
            real_D = self.discriminator(dis_target_inputs, training=True)
            fake_D = self.discriminator(dis_output_inputs, training=True)
            # real_D = self.discriminator(targets, training=True)
            # fake_D = self.discriminator(outputs, training=True)

            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_D, labels=tf.ones_like(real_D)
            ))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_D, labels=tf.zeros_like(fake_D)
            ))
            d_loss = d_loss_real + d_loss_fake

            cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_D, labels=tf.ones_like(fake_D)
            ))
            g_loss = emd_loss * self.args.L1_weight + cheat_loss

            loss_dict['d_loss_real'] = d_loss_real / self.num_gpu
            loss_dict['d_loss_fake'] = d_loss_fake / self.num_gpu
            loss_dict['d_loss'] = d_loss / self.num_gpu
            loss_dict['cheat_loss'] = cheat_loss / self.num_gpu
            loss_dict['g_loss'] = g_loss / self.num_gpu
        else:
            loss_dict['g_loss'] = emd_loss / self.num_gpu

        return loss_dict

################################################################################
# multi style
def concat_data(inputs, batch_size, style_num):
    outputs = []
    for k in range(style_num):
        sample_L = []
        for i in range(batch_size):
            begin = i * style_num + k
            sample_L.append(inputs[begin:begin+1, :, :, :])
        outputs.append(tf.concat(sample_L, 0))
    return outputs

class Mixer_multi_style(layers.Layer):
    def __init__(self, out_dim):
        super(Mixer_multi_style, self).__init__()
        self.out_dim = out_dim
    
    def build(self, input_shape):
        w_init = tf.random_normal_initializer(0, 0.02)
        self.w = tf.Variable(
            name='w',
            initial_value=w_init(shape=(input_shape[0][-1], input_shape[1][-1], 
                self.out_dim), dtype='float32'),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            name='b',
            initial_value=b_init(shape=(self.out_dim,), dtype='float32'),
            trainable=True
        )
    
    def call(self, inputs):
        input_s = inputs[0]
        input_c = inputs[1]
        # [style_num, b, 512] * [512, 512, 512] -> [style_num, b, 512, 512]
        outputs = tf.einsum('abi,ijk->abjk', input_s, self.w)
        # [b, 512] * [style_num, b, 512, 512] -> [style_num, b, 512]
        outputs = tf.einsum('bj,abjk->abk', input_c, outputs)
        return outputs + self.b

class EMD_multi_style(keras.Model):
    def __init__(self, batch_size, style_num, input_size=80, return_penultimate=False):
        super(EMD_multi_style, self).__init__()
        self.batch_size = batch_size
        self.style_num = style_num
        self.input_size = input_size
        self.return_penultimate = return_penultimate
        for i in range(style_num):
            setattr(self, 's_encoder_%d' % i, Encoder(input_size))
        self.c_encoder = Encoder(input_size)
        self.mix_out = 512
        self.mixer = Mixer_multi_style(self.mix_out)
        self.decoder = Decoder(input_size)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        # [b * style_num, 80, 80, 10] -> list (length=style_num)
        input_s_list = concat_data(inputs[0], self.batch_size, self.style_num)
        input_c = inputs[1]
        s_list = []
        for i in range(self.style_num):
            s_outputs = getattr(self, 's_encoder_%d' % i)(input_s_list[i])
            s = self.flatten(s_outputs[-1])
            s_list.append(s[None])
        # styles: [style_num, b, 512]
        styles = tf.concat(s_list, 0)
        
        # average style features
        # style_sum = s_list[0]
        # for i in range(1, len(s_list)):
        #     style_sum += s_list[i]
        # styles = tf.concat([style_sum / len(s_list) for _ in s_list], 0)
        c_outputs = self.c_encoder(input_c)
        c = self.flatten(c_outputs[-1])
        
        mix = self.mixer([styles, c])
        b, h, w, _ = c_outputs[-1].get_shape()
        mix_list = []
        for i in range(self.style_num):
            # use tf.reshape will raise error
            mix_reshape = layers.Reshape([h, w, self.mix_out])(mix[i])
            mix_list.append(mix_reshape)
        # mix: [b * style_num, 1, 1, 512]
        mix = tf.concat(mix_list, 0)
        
        for i in range(len(c_outputs)):
            c_list = []
            for j in range(self.style_num):
                c_list.append(c_outputs[i])
            # [b * style_num, h, w, c]
            c_outputs[i] = tf.concat(c_list, 0)
        outputs = self.decoder(mix, c_outputs)
        if self.return_penultimate:
            return outputs[-2]
        else:
            return tf.sigmoid(outputs[-1])
        
    def interpolate(self, inputs, args):
        # [b * style_num, 80, 80, 10] -> list (length=style_num)
        input_s = concat_data(inputs[0], self.batch_size, self.style_num)
        input_c = inputs[1]
        s_list = []
        for i in range(self.style_num):
            s_outputs = getattr(self, 's_encoder_%d' % i)(input_s[i], training=False)
            s = self.flatten(s_outputs[-1])
            s_list.append(s[None])
        # styles: [style_num, b, 512]
        styles = tf.concat(s_list, 0)
        c_outputs = self.c_encoder(input_c, training=False)
        c = self.flatten(c_outputs[-1])
        b, h, w, _ = c_outputs[-1].get_shape()
        for i in range(len(c_outputs)):
            c_list = []
            for j in range(self.style_num):
                c_list.append(c_outputs[i])
            # [b * style_num, h, w, c]
            c_outputs[i] = tf.concat(c_list, 0)

        left, right = args.interp_interval
        interp_len = right - left
        n = args.interp_num
        alphas = np.arange(left, right + interp_len / n, interp_len / n)
        interp_outputs = []
        for alpha in alphas:
            style_01 = (1 - alpha) * styles[0] + alpha * styles[1]
            style_02 = (1 - alpha) * styles[0] + alpha * styles[2]
            style_12 = (1 - alpha) * styles[1] + alpha * styles[2]
            new_styles = tf.concat([style_01[None], style_02[None], style_12[None]], 0)
        
            mix = self.mixer([new_styles, c])
            mix_list = []
            for i in range(self.style_num):
                # use tf.reshape will raise error
                mix_reshape = layers.Reshape([h, w, self.mix_out])(mix[i])
                mix_list.append(mix_reshape)
            # mix: [b * style_num, 1, 1, 512]
            mix = tf.concat(mix_list, 0)
            outputs = self.decoder(mix, c_outputs, training=False)
            if self.return_penultimate:
                interp_outputs.append(outputs[-2])
            else:
                interp_outputs.append(tf.sigmoid(outputs[-1]))
        return interp_outputs
    
    def interpolate_3style(self, inputs, args, coeff=None):
        if not isinstance(coeff, np.ndarray):
            # default coefficient, along three medians of the triangle
            left, right = args.interp_interval
            interp_len = right - left
            n = args.interp_num
            alphas = np.arange(left, right + interp_len / n, interp_len / n)
            coeff = np.concatenate(
                [1 - alphas[:, np.newaxis], 0.5 * alphas[:, np.newaxis], 0.5 * alphas[:, np.newaxis]], 1)
        # [b * style_num, 80, 80, 10] -> list (length=style_num)
        input_s = concat_data(inputs[0], self.batch_size, self.style_num)
        input_c = inputs[1]
        s_list = []
        for i in range(self.style_num):
            s_outputs = getattr(self, 's_encoder_%d' % i)(input_s[i])
            s = self.flatten(s_outputs[-1])
            s_list.append(s[None])
        # styles: [style_num, b, 512]
        styles = tf.concat(s_list, 0)
        c_outputs = self.c_encoder(input_c)
        c = self.flatten(c_outputs[-1])
        b, h, w, _ = c_outputs[-1].get_shape()
        for i in range(len(c_outputs)):
            c_list = []
            for j in range(self.style_num):
                c_list.append(c_outputs[i])
            # [b * style_num, h, w, c]
            c_outputs[i] = tf.concat(c_list, 0)

        interp_outputs = []
        for i in range(coeff.shape[0]):
            alpha, beta, gamma = coeff[i]
            style_012 = alpha * styles[0] + beta * styles[1] + gamma * styles[2]
            style_102 = alpha * styles[1] + beta * styles[0] + gamma * styles[2]
            style_201 = alpha * styles[2] + beta * styles[0] + gamma * styles[1]
            new_styles = tf.concat([style_012[None], style_102[None], style_201[None]], 0)
        
            mix = self.mixer([new_styles, c])
            mix_list = []
            for i in range(self.style_num):
                # use tf.reshape will raise error
                mix_reshape = layers.Reshape([h, w, self.mix_out])(mix[i])
                mix_list.append(mix_reshape)
            # mix: [b * style_num, 1, 1, 512]
            mix = tf.concat(mix_list, 0)
            outputs = self.decoder(mix, c_outputs)
            if self.return_penultimate:
                interp_outputs.append(outputs[-2])
            else:
                interp_outputs.append(tf.sigmoid(outputs[-1]))
        return interp_outputs

################################################################################
# local enhancer
class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

class resnet_block(keras.Model):
    def __init__(self, out_channel):
        super(resnet_block, self).__init__()
        self.resnet_block = keras.Sequential([
            ReflectionPadding2D((1, 1)),
            layers.Conv2D(out_channel, kernel_size=3, strides=1, padding='valid'),
            layers.BatchNormalization(), layers.ReLU(),
            ReflectionPadding2D((1, 1)),
            layers.Conv2D(out_channel, kernel_size=3, strides=1, padding='valid'),
            layers.BatchNormalization()
        ])
    
    def call(self, x):
        return x + self.resnet_block(x)

def downsample(channel):
    result = models.Sequential([
        ReflectionPadding2D((2, 2)),
        layers.Conv2D(channel, kernel_size=5, strides=1, padding='valid'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(channel*2, kernel_size=7, strides=3, padding='valid'),
        layers.BatchNormalization(), layers.ReLU(),
        ReflectionPadding2D((2, 2)),
        layers.Conv2D(channel*2, kernel_size=5, strides=1, padding='valid'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(channel*2, kernel_size=5, strides=1, padding='valid'),
        layers.BatchNormalization(), layers.ReLU(),
    ])
    return result

def upsample(channel):
    result = models.Sequential([
        resnet_block(channel*2), resnet_block(channel*2), resnet_block(channel*2),
        layers.Conv2DTranspose(channel*2, kernel_size=5, strides=1, padding='valid'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(channel, kernel_size=7, strides=3, padding='valid'),
        layers.BatchNormalization(), layers.ReLU(),
        ReflectionPadding2D((3, 3)),
        layers.Conv2D(1, kernel_size=7, strides=1, padding='valid')
    ])
    return result

class local_enhancer(keras.Model):
    def __init__(self):
        super(local_enhancer, self).__init__()
        self.global_net = EMD(return_penultimate=True)
        self.avg_pool = layers.AveragePooling2D(pool_size=19, strides=3, padding='valid')
        channel = 32
        self.s_downsample = downsample(channel)
        self.c_downsample = downsample(channel)
        self.upsample = upsample(channel)
    
    def call(self, inputs):
        input_s = inputs[0]
        input_c = inputs[1]
        global_input_s = self.avg_pool(input_s)
        global_input_c = self.avg_pool(input_c)
        global_outputs = self.global_net([global_input_s, global_input_c])

        s_local = self.s_downsample(input_s)
        c_local = self.c_downsample(input_c)
        outputs = 0.3 * c_local + 0.3 * s_local + global_outputs
        outputs = self.upsample(outputs)

        return tf.sigmoid(outputs)
    
    def summary(self):
        x1 = layers.Input(shape=(256, 256, 10))
        x2 = layers.Input(shape=(256, 256, 10))
        model = keras.Model(inputs=[x1, x2], outputs=self.call([x1, x2]))
        return model.summary()

class local_enhancer_multi_style(keras.Model):
    def __init__(self, batch_size, style_num):
        super(local_enhancer_multi_style, self).__init__()
        self.style_num = style_num
        self.batch_size = batch_size
        self.global_net = EMD_multi_style(batch_size, style_num, return_penultimate=True)
        self.avg_pool = layers.AveragePooling2D(pool_size=19, strides=3, padding='valid')
        channel = 32
        for i in range(style_num):
            setattr(self, 's_downsample_%d' % i, downsample(channel))
        self.c_downsample = downsample(channel)
        self.upsample = upsample(channel)
    
    def call(self, inputs):
        input_s = inputs[0]
        input_c = inputs[1]
        global_input_s = self.avg_pool(input_s)
        global_input_c = self.avg_pool(input_c)
        global_outputs = self.global_net([global_input_s, global_input_c])

        input_s_list = concat_data(input_s, self.batch_size, self.style_num)
        c_local = self.c_downsample(input_c)
        c_local_list = []
        s_local_list = []
        for i in range(self.style_num):
            c_local_list.append(c_local)
            s_local = getattr(self, 's_downsample_%d' % i)(input_s_list[i])
            s_local_list.append(s_local)
        c_local = tf.concat(c_local_list, 0)
        s_local = tf.concat(s_local_list, 0)
        outputs = 0.3 * c_local + 0.3 * s_local + global_outputs
        outputs = self.upsample(outputs)

        return tf.sigmoid(outputs)
    
    def interpolate(self, inputs, args):
        input_s = inputs[0]
        input_c = inputs[1]
        global_input_s = self.avg_pool(input_s)
        global_input_c = self.avg_pool(input_c)
        global_outputs = self.global_net.interpolate([global_input_s, global_input_c], args)

        input_s_list = concat_data(input_s, self.batch_size, self.style_num)
        c_local = self.c_downsample(input_c)
        c_local_list = []
        s_local_list = []
        for i in range(self.style_num):
            c_local_list.append(c_local)
            s_local = getattr(self, 's_downsample_%d' % i)(input_s_list[i])
            s_local_list.append(s_local)
        c_local = tf.concat(c_local_list, 0)
        
        left, right = args.interp_interval
        interp_len = right - left
        n = args.interp_num
        alphas = np.arange(left, right + interp_len / n, interp_len / n)
        global_interp_outputs = []
        for i in range(len(alphas)):
            style_local_01 = (1 - alphas[i]) * s_local_list[0] + alphas[i] * s_local_list[1]
            style_local_02 = (1 - alphas[i]) * s_local_list[0] + alphas[i] * s_local_list[2]
            style_local_12 = (1 - alphas[i]) * s_local_list[1] + alphas[i] * s_local_list[2]
            new_s_local = tf.concat([style_local_01, style_local_02, style_local_12], 0)
            
            outputs = 0.3 * c_local + 0.3 * new_s_local + global_outputs[i]
            outputs = self.upsample(outputs)
            global_interp_outputs.append(tf.sigmoid(outputs))

        return global_interp_outputs
    
    def interpolate_3style(self, inputs, args):
        left, right = args.interp_interval
        interp_len = right - left
        n = args.interp_num
        alphas = np.arange(left, right + interp_len / n, interp_len / n)
        coeff = np.concatenate(
            [1 - alphas[:, np.newaxis], 0.5 * alphas[:, np.newaxis], 0.5 * alphas[:, np.newaxis]], 1)

        # points = np.array([[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [-0.1, 0], [-0.2, 0], [-0.3, 0], [-0.4, 0], [-0.5, 0], 
        #     [-0.05, 0.779], [-0.1, 0.692], [-0.15, 0.606], [-0.2, 0.519], [-0.25, 0.433], [-0.3, 0.346], [-0.35, 0.259], [-0.4, 0.173], [-0.45, 0.086], 
        #     [0.05, 0.779], [0.1, 0.692], [0.15, 0.606], [0.2, 0.519], [0.25, 0.433], [0.3, 0.346], [0.35, 0.259], [0.4, 0.173], [0.45, 0.086], 
        #     [-0.1, 0.1], [-0.1, 0.2], [-0.1, 0.3], [-0.1, 0.4], [-0.1, 0.5], [-0.1, 0.6], [-0.2, 0.1], [-0.2, 0.2], [-0.2, 0.3], [-0.2, 0.4], [-0.3, 0.1], [-0.3, 0.2], [-0.4, 0.1], 
        #     [0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4], [0.1, 0.5], [0.1, 0.6], [0.2, 0.1], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.3, 0.1], [0.3, 0.2], [0.4, 0.1], 
        #     [0, 0.1], [0, 0.2], [0, 0.3], [0, 0.4], [0, 0.5], [0, 0.6], [0, 0.7], [0, 0.866]])
        # ones_array = np.ones((points.shape[0], 1))
        # points = np.expand_dims(np.concatenate([points, ones_array], 1), axis=2)
        # A = np.array([[0, -1/2, 1/2], [np.sqrt(3)/2, 0, 0], [1, 1, 1]])
        # coeff = np.linalg.inv(A) @ points
        
        input_s = inputs[0]
        input_c = inputs[1]
        global_input_s = self.avg_pool(input_s)
        global_input_c = self.avg_pool(input_c)
        global_outputs = self.global_net.interpolate_3style([global_input_s, global_input_c], args, coeff)

        input_s_list = concat_data(input_s, self.batch_size, self.style_num)
        c_local = self.c_downsample(input_c)
        c_local_list = []
        s_local_list = []
        for i in range(self.style_num):
            c_local_list.append(c_local)
            s_local = getattr(self, 's_downsample_%d' % i)(input_s_list[i])
            s_local_list.append(s_local)
        c_local = tf.concat(c_local_list, 0)
        
        global_interp_outputs = []
        for i in range(coeff.shape[0]):
            alpha, beta, gamma = coeff[i]
            style_local_012 = alpha * s_local_list[0] + beta * s_local_list[1] + gamma * s_local_list[2]
            style_local_102 = alpha * s_local_list[1] + beta * s_local_list[0] + gamma * s_local_list[2]
            style_local_201 = alpha * s_local_list[2] + beta * s_local_list[0] + gamma * s_local_list[1]
            new_s_local = tf.concat([style_local_012, style_local_102, style_local_201], 0)
            
            outputs = 0.3 * c_local + 0.3 * new_s_local + global_outputs[i]
            outputs = self.upsample(outputs)
            global_interp_outputs.append(tf.sigmoid(outputs))

        return global_interp_outputs
        

# model = Encoder()
# model.summary()