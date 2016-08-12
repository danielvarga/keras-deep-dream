# Keras deep dream

Keras deep dream implementation, based on
https://github.com/fchollet/keras/blob/master/examples/deep_dream.py,
with two new features:

- the ability to choose any pre-trained network from https://github.com/fchollet/deep-learning-models for dreaming
- implementing the 'octaves' iterative re-scaling functionality that was missing from the Keras example.


Usage:

https://github.com/fchollet/deep-learning-models is not (yet) a python module, so the easiest way
to set things up is to clone that repo, copy or softlink  ```deep_dream.py``` from here to there,
and running it there.

```
python deep_dream.py [ vgg16 | vgg19 | resnet50 | inception_v3 ] input.png output_filename_prefix [ layer ]
```

Like in the original example, most of the parameter tuning happens
in the ```setting``` dict variable in the source code. If the
optional positional ```layer``` argument is not provided,
it's taken from ```settings['features']````.


# Caveat

The algorithm compiles a new Keras model for each image size, and for larger networks like resnet50,
compilation is very slow with the Theano backend. Does anyone know how to avoid this?
In Caffe it's just https://github.com/google/deepdream/blame/master/dream.ipynb#L208


# Backstory

The deep dream algorithm is quite a bit 2015 now, but I always wanted to see it running on residual networks,
and with https://github.com/fchollet/deep-learning-models, I've grabbed the opportunity.

Porting https://github.com/fchollet/keras/blob/master/examples/deep_dream.py
to work with ResNet50 was trivial, but the results were quite boring.
The Keras deep dream example does not implement the octaves functionality
of the original deep dream algorithm, which goes through iterative re-scaling
of the image, and it helps tremendously large-scale dreamed objects to appear.
So I've implemented octaves, hoping that it would make deep dreaming on resnets more interesting.
Unfortunately, it did not help much. But this is the first full deep dream
implementation on Keras that I'm aware of, so at least that's something.
I haven't yet given up on using deep dream to help understand resnets better,
story in progress.
