from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Lambda, BatchNormalization, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, ZeroPadding2D, merge, add
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, Callback, callbacks
from keras.preprocessing.image import ImageDataGenerator

import os, glob, sys, threading
import scipy.io
import h5py
from PIL import Image

from tifffile import imread
import matplotlib.pyplot as plt

import time
import numpy
import re
import math
import argparse

from keras.backend.tensorflow_backend import set_session

import signal

def SubpixelConv2D(scale, **kwargs):
    import tensorflow as tf
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def get_image_batch(h5_file_path, offset, batch_size):

    print('Reading file {}, offset {}'.format(h5_file_path, offset), end='\r')
    sys.stdout.write("\033[K") #clear line
    with h5py.File(h5_file_path, 'r') as h5fd:
        shape = h5fd['data'].shape
        data = numpy.array(h5fd['data'][offset:offset+batch_size])
        label = numpy.array(h5fd['label'][offset:offset+batch_size])

    return data, label, shape

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def image_gen(target_list, batch_size):
    offset = 0
    target_count = 0
    while True:
        for target in target_list:
            batch_x, batch_y, shape = get_image_batch(target, offset, batch_size)
            if target_count == len(target_list):
                #offset += batch_size
                offset = numpy.random.randint(shape[0]-batch_size)
                target_count = 0
            if offset >= shape[0]:
                offset = numpy.random.randint(shape[0]-batch_size)
            target_count += 1
            yield (batch_x, batch_y)

def tf_log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

# SSIM loss function
def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def ssim_metric(y_true, y_pred):
    # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    patches_true = tf.image.extract_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.image.extract_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.where(tf.math.is_nan(ssim), K.zeros_like(ssim), ssim)
    return ssim

SummaryDir=os.path.join(os.getcwd(), 'summaries/CNN_nobgremov')
BatchNum=64
InputDimension=[200,200,1]
GtDimensions=[400,400,1]
LearningRate=1e-4
NumIteration=100000
NumTrainImages=5680
NumKernels = [16,32,32,64,64,4]
FilterSizes =  [3,3,3,3,3,3]

InputData = tf.placeholder(tf.float32, [BatchNum]+InputDimension) #network input
InputGT = tf.placeholder(tf.float32, [BatchNum]+GtDimensions) #network input

class MyCallback(Callback):
    
    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print('Learning rate with decay: {}'.format(K.eval(lr_with_decay)))
        print('Decay: {}'.format(K.eval(decay)))
        print('Initial learning rate: {}'.format(K.eval(lr)))

def MakeConvNet(Size, batch_size, epochs, optimizer, learning_rate, train_list, validation_list):
    input_img = Input(shape=Size)
    model = input_img
    CurrentInput = InputData
    Channels = Size[2] #the input dim at the first layer is 1, since the input image is grayscale

    for i in range(len(NumKernels)-1): #number of layers
        NumKernel=NumKernels[i]
        FilterSize = FilterSizes[i]
        print(i)

        model = Conv2DTranspose(NumKernel, (FilterSize, FilterSize), padding='same', kernel_initializer='he_normal', use_bias=False)(model)
        print(model.shape)
        model = BatchNormalization()(model)
        print(model.shape)
        # model = Activation('relu')(model)
        model = LeakyReLU()(model)
        print(model.shape)

    model = Conv2DTranspose(4, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(model)
    print(model.shape)
    model = BatchNormalization()(model)
    print(model.shape)
    # model = Activation('relu')(model)
    model = SubpixelConv2D(2)(model)
    model = LeakyReLU()(model)
    print(model.shape)
    # print(input_img.shape)
    model = Model(input_img, model)

    adam = Adadelta()
    sgd = SGD(lr=learning_rate, momentum=0.9, decay=1e-4, nesterov=False, clipnorm=1)
    if optimizer == 0:
        model.compile(adam, loss='mean_absolute_error', metrics=[ssim, ssim_metric, PSNR])
    else:
        model.compile(sgd, loss='mean_absolute_error', metrics=[ssim, ssim_metric, PSNR])

    model.summary()

    mycallback = MyCallback(model)
    timestamp = time.strftime("%m%d-%H%M", time.localtime(time.time()))

    csv_logger = callbacks.CSVLogger('data/callbacks/training_{}.log'.format(timestamp))
    filepath="./checkpoints/weights-{epoch:03d}-{PSNR:.2f}-{ssim:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')
    callbacks_list = [mycallback, checkpoint, csv_logger]
    
    with open('./model/subcnn_architecture_{}.json'.format(timestamp), 'w') as f:
        f.write(model.to_json())

    history = model.fit_generator(image_gen(train_list, batch_size=batch_size), 
                        steps_per_epoch=(3600)*len(train_list) // batch_size,
                        validation_data=image_gen(validation_list,batch_size=batch_size),
                        validation_steps=(3600)*len(validation_list) // batch_size,
                        epochs=epochs,
                        workers=1024,
                        callbacks=callbacks_list,
                        verbose=1)

    print("Done training!!!")
    
    # return Out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train a SubCNN.')
    parser.add_argument('-s', '--size', help='Size of one dimension of the square image.', required=True, type=int)
    parser.add_argument('-b', '--batch-size', help='Batch size.', required=True, type=int, default=64)
    parser.add_argument('-e', '--epochs', help='Amount of epochs.', required=True, type=int, default=100)
    parser.add_argument('-o', '--optimizer', help='0: Adam, 1: SGD.', required=True, type=int, choices={0, 1}, default=0)
    parser.add_argument('-l', '--lr', help='Learning rate value.', required=True, type=float, default=0.00001)
    parser.add_argument('-t', '--train-list', help='List of H5 files paths where the training data is located.', required=True)
    parser.add_argument('-v', '--validation-list', help='List of H5 files paths where the validation data is located.', required=True)

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    arguments = parser.parse_args()

    try:
        with open(arguments.train_list) as tfd:
            train_list = []
            for l in tfd.readlines():
                train_list.append(l.strip())
    except Exception as e:
        print('Insame training file list', file=sys.stderr)
        sys.exit(e)
    
    try:
        with open(arguments.validation_list) as vfd:
            validation_list = []
            for l in vfd.readlines():
                validation_list.append(l.strip())
    except Exception as e:
        print('Insame validation file list', file=sys.stderr)
        sys.exit(e)

    batch_size = arguments.batch_size
    size = (arguments.size, arguments.size, 1)
    epochs = arguments.epochs
    optimizer = arguments.optimizer
    learning_rate = arguments.lr

    MakeConvNet(size, batch_size, epochs, optimizer, learning_rate, train_list, validation_list)
