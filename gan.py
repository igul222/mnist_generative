"""Generative Adversarial Network for MNIST."""

import os, sys
sys.path.append(os.getcwd())

import swft
import lasagne
import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import functools

BATCH_SIZE = 100
EPOCHS = 100

theano_srng = RandomStreams(seed=234)

def ReLULayer(name, n_in, n_out, inputs):
    output = swft.ops.Linear(name+'.Linear', n_in, n_out, inputs, initialization=('uniform', 0.05))
    output = swft.ops.BatchNormalize(name+'.BN', n_out, output)
    return swft.ops.rectify(output)

def MaxoutLayer(name, n_in, n_out, inputs):
    PIECE_SIZE = 5
    output = swft.ops.Linear(name+'.Linear', n_in, n_out * PIECE_SIZE, inputs, initialization=('uniform', 0.005))
    units = output.reshape((
        inputs.shape[0], 
        n_out, 
        PIECE_SIZE
    ))
    return T.max(units, axis=2)

def generator(n_samples):
    noise = theano_srng.uniform(
        size=(n_samples, 100), 
        low=-swft.floatX(numpy.sqrt(3)),
        high=swft.floatX(numpy.sqrt(3))
    )

    output = ReLULayer('Generator.1', 100, 1200, noise)
    output = ReLULayer('Generator.2', 1200, 1200, output)
    output = ReLULayer('Generator.3', 1200, 1200, output)
    output = ReLULayer('Generator.4', 1200, 1200, output)
    
    return T.nnet.sigmoid(
        swft.ops.Linear('Generator.5', 1200, 784, output, initialization=('uniform', 0.05))
    )

def discriminator(inputs, debug=False):
    inputs = swft.ops.Dropout(0.2, inputs)
    output = MaxoutLayer('Discriminator.1', 784, 240, inputs)
    output = swft.ops.Dropout(0.5, output)
    output = MaxoutLayer('Discriminator.2', 240, 240, output)
    output = swft.ops.Dropout(0.5, output)

    # We apply the sigmoid in a later step
    return swft.ops.Linear('Discriminator.Output', 240, 1, output, initialization=('uniform', 0.005)).flatten()

symbolic_inputs = swft.mnist.symbolic_inputs()
images, targets = symbolic_inputs

generator_output = generator(BATCH_SIZE)

disc_out = discriminator(T.concatenate([generator_output, images], axis=0))
disc_gen_out = T.nnet.sigmoid(disc_out[:BATCH_SIZE])
disc_inputs  = T.nnet.sigmoid(disc_out[BATCH_SIZE:])

# Gen objective:  push D(G) to one
gen_cost      = T.nnet.binary_crossentropy(disc_gen_out, swft.floatX(1)).mean()
gen_cost.name = 'gen_cost'

# Discrim objective: push D(G) to zero, and push D(real) to one
discrim_cost  = T.nnet.binary_crossentropy(disc_gen_out, swft.floatX(0)).mean()
discrim_cost += T.nnet.binary_crossentropy(disc_inputs, swft.floatX(1)).mean()
discrim_cost /= swft.floatX(2.0)
discrim_cost.name = 'discrim_cost'

train_data, dev_data, test_data = swft.mnist.load(BATCH_SIZE)

gen_params     = swft.search(gen_cost,     lambda x: hasattr(x, 'param') and 'Generator' in x.name)
discrim_params = swft.search(discrim_cost, lambda x: hasattr(x, 'param') and 'Discriminator' in x.name)

_sample_fn = theano.function([], generator(100))
def generate_image(epoch):
    sample = _sample_fn()
    # the transpose is rowx, rowy, height, width -> rowy, height, rowx, width
    sample = sample.reshape((10,10,28,28)).transpose(1,2,0,3).reshape((10*28, 10*28))
    plt.imshow(sample, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.savefig('epoch'+str(epoch))

swft.train(
    symbolic_inputs,
    [gen_cost, discrim_cost],
    train_data,
    dev_data=dev_data,
    param_sets = [gen_params, discrim_params],
    optimizers=[
        functools.partial(lasagne.updates.momentum, learning_rate=0.1, momentum=0.5),
        functools.partial(lasagne.updates.momentum, learning_rate=0.1, momentum=0.5)
    ],
    epochs=EPOCHS,
    print_every=1000,
    callback=generate_image
)