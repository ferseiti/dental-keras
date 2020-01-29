from keras.models import load_model, model_from_json, Model
from tifffile import imsave, imread
import argparse
import tensorflow as tf
import json
from SubCNN_keras_train import *

def predict_subcnn(model, img):

    upsampled = model.predict(img)
    imsave('predicted.tiff', upsampled)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Predict with SubCNN.')
    parser.add_argument('-j', '--json-path', help='Path where the model json is located.', required=True)
    parser.add_argument('-w', '--weights-path', help='Path where the model is located.', required=True)
    parser.add_argument('-i', '--image-path', help='Path where the image is located.', required=True)
    # parser.add_argument('-o', '--output-path', help='Path where the output image will be saved.', required=True)
    # parser.add_argument('-s', '--size', help='Size (width) of the square image', required=True, type=int)

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    arguments = parser.parse_args()

    # model = model_from_json(json.loads(arguments.json_path))
    
    model = load_model(arguments.weights_path, custom_objects={'tf': tf, 'ssim': ssim, 'ssim_metric': ssim_metric, 'PSNR': PSNR})
    model.layers.pop(0)
    

    img = imread(arguments.image_path)

    newInput = Input(batch_shape=((1,)+img.shape + (1,)))    # let us say this new InputLayer
    newOutputs = model(newInput)
    newModel = Model(newInput, newOutputs)

    newModel.summary()
    model.summary()

    img = img.reshape((1,)+ img.shape +(1,))
    predict_subcnn(newModel, img)
