'''Deep Dreaming in Keras.

Run the script with:
```
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
e.g.:
```
python deep_dream.py img/mypic.jpg results/dream
```

It is preferrable to run this script on GPU, for speed.
If running on CPU, prefer the TensorFlow backend (much faster).

Example results: http://i.imgur.com/FX6ROg9.jpg
'''
from __future__ import print_function
from scipy.misc import imread, imresize, imsave
import scipy.ndimage
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import h5py
import os

from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Input
from keras import backend as K

import vgg16
import vgg19
import resnet50
import inception_v3

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('model', metavar='model', type=str,
                    help='The model, either vgg16 or resnet50')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('layer', metavar='layer', type=str, nargs='?', default=None,
                    help='Layer to dream.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix


# some settings we found interesting
saved_settings = {
    'bad_trip': {'features': {'block4_conv1': 0.05,
                              'block4_conv2': 0.01,
                              'block4_conv3': 0.01},
                 'continuity': 0.1,
                 'dream_l2': 0.8,
                 'jitter': 5},
    'dreamy': {'features': {'conv5_1': 0.05,
                            'conv5_2': 0.02},
               'continuity': 0.1,
               'dream_l2': 0.02,
               'jitter': 0},
    'daniel': {'features': {'conv5_1': 0.05},
               'continuity': 0.1,
               'dream_l2': 1.0,
               'jitter': 1,
               'neuron_id': 21}, # 21 textile, 26 eyes
    'resnet': {'features': {'bn4e_branch2b': 1.0},
                 'continuity': 0.0,
                 'dream_l2': 0.0,
                 'jitter': 5,
                 'octave_count': 4,
                 'octave_scale': 1.2}
}

# the settings we will use in this experiment
settings = saved_settings['resnet']

# The command line overrides the settings:
if args.layer is not None:
    settings['features'] = { args.layer : 1.0 }


# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(img):
    img = img.transpose((2, 0, 1)).astype('float64')
    img = img[:3, :, :] # getting rid of alpha.
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# continuity loss util function
def continuity_loss(x, img_shape):
    # TODO The names are bad, reversed, the action is good.
    h, w = img_shape
    assert K.ndim(x) == 4
    a = K.square(x[:, :, :h-1, :w-1] - x[:, :, 1:, :w-1])
    b = K.square(x[:, :, :h-1, :w-1] - x[:, :, :h-1, 1:])
    return K.sum(K.pow(a + b, 1.25))

def create_loss_function(dream, settings, model, img_shape):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    img_height, img_width = img_shape
    # define the loss
    loss = K.variable(0.)
    features = settings['features']
    for layer_name in features:
        # add the L2 norm of the features of a layer to the loss
        assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
        coeff = features[layer_name]
        x = layer_dict[layer_name].output
        shape = layer_dict[layer_name].output_shape
        print("adding layer %s with output shape %s" % (layer_name, shape))
        # we avoid border artifacts by only involving non-border pixels in the loss
        loss -= coeff * shape[1] * K.sum(K.square(x[:, :, 2: shape[2]-2, 2: shape[3]-2])) / np.prod(shape[1:])

    # add continuity loss (gives image local coherence, can result in an artful blur)
    loss += settings['continuity'] * continuity_loss(dream, img_shape) / (3 * img_width * img_height)
    # add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
    loss += settings['dream_l2'] * K.sum(K.square(dream)) / (3 * img_width * img_height)

    # feel free to further modify the loss as you see fit, to achieve new effects...

    # compute the gradients of the dream wrt the loss
    grads = K.gradients(loss, dream)

    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs_before_currying = K.function([dream, K.learning_phase()], outputs)
    f_outputs = lambda dream: f_outputs_before_currying([dream, 0]) # 0=test

    def loss_and_grads(x):
        x = x.reshape((1, 3, img_shape[0], img_shape[1]))
        outs = f_outputs(x)
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    return loss_and_grads

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self, loss_and_grads):
        self.loss_value = None
        self.grads_values = None
        self.loss_and_grads = loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


img = imread(base_image_path)

octave_count = settings['octave_count']
octave_scale = settings['octave_scale']
octaves = [img]
for _ in range(1, octave_count):
    octaves.append(scipy.ndimage.zoom(octaves[-1], (1.0/octave_scale, 1.0/octave_scale, 1), order=1))
    print(octaves[-1].shape)

detail = np.zeros_like(octaves[-1])

for octave_index, octave_base in enumerate(octaves[::-1]):
    img_shape = octave_base.shape[:2]
    img_height, img_width = img_shape
    print('Starting octave %d with dimensions %d x %d' % (octave_index, img_width, img_height))

    if octave_index > 0:
        # upscale details from the previous octave
        detail_height, detail_width = detail.shape[:2]
        # detail = scipy.ndimage.zoom(detail, (1.0/octave_scale, 1.0/octave_scale, 1), order=1)
        print('resizing detail from %s to %s' % (detail.shape, octave_base.shape))
        detail = imresize(detail, octave_base.shape[:2])

    x = preprocess_image(octave_base + detail)

    dream = Input(shape=(3, img_shape[0], img_shape[1]))

    if args.model=='resnet50':
        model = resnet50.ResNet50(include_top=False, input_tensor=dream)
    elif args.model=='vgg16':
        model = vgg16.VGG16(include_top=False, input_tensor=dream)
    elif args.model=='vgg19':
        model = vgg19.VGG19(include_top=False, input_tensor=dream)
    elif args.model=='inception_v3':
        model = inception_v3.InceptionV3(include_top=False, input_tensor=dream)
    else:
        raise 'unknown model '+args.model
    print('Model loaded.')

    loss_and_grads = create_loss_function(dream, settings, model, img_shape)
    evaluator = Evaluator(loss_and_grads)

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the loss
    for i in range(10):
        print('Start of iteration', i)
        start_time = time.time()

        # add a random jitter to the initial image. This will be reverted at decoding time
        random_jitter = (settings['jitter'] * 2) * (np.random.random(x.shape) - 0.5)
        x += random_jitter

        # run L-BFGS for 7 steps
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=7)
        print('Current loss value:', min_val)
        # decode the dream and save it
        x = x.reshape((1, 3, img_height, img_width))
        x -= random_jitter
        img = deprocess_image(x[0])
        fname = result_prefix + '_at_iteration_%d_%d.png' % (octave_index, i)
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
