"""
`evaluate(...)` measures the quality of samples from an MNIST generative model
by training a discriminator to distinguish the samples from real MNIST images.
The returned score is the cost of that discriminator on a held-out validation
set (the higher the discriminator's cost, the better the generative model).

Note: you need to pass exactly 60K samples to `evaluate(...)` (50K for training,
10K for validation).

Right now the discriminator is a shallow MLP. In theory it can be anything, but
you'll get best results if it's significantly weaker than the generative model.

Unfortunately right now even this (pretty weak) discriminator has gets >95 
percent accuracy on both generative models. TODO: Try simpler discriminators 
(logistic regression, kNN?)
"""

import swft
import numpy
import theano
import theano.tensor as T

import functools

INPUT_DIM = 784
HIDDEN_DIM = 1024
BATCH_SIZE = 100
EPOCHS = 15

def _Layer(name, n_in, n_out, inputs):
    return swft.ops.rectify(
        swft.ops.Linear(name, n_in, n_out, inputs)
    )

def _evaluator(image):
    output = _Layer('Evaluator.1', INPUT_DIM, HIDDEN_DIM, image)
    output = _Layer('Evaluator.2', HIDDEN_DIM, HIDDEN_DIM, output)
    return T.nnet.sigmoid(
        swft.ops.Linear('Evaluator.Output', HIDDEN_DIM, 1, output).flatten()
    )

def evaluate(fakes):
    real_images = T.matrix()
    fake_images = T.matrix()

    cost  = T.nnet.binary_crossentropy(_evaluator(real_images), swft.floatX(1)).mean()
    cost += T.nnet.binary_crossentropy(_evaluator(fake_images), swft.floatX(0)).mean()

    real_accuracy = T.ge(_evaluator(real_images), swft.floatX(0.5)).mean()
    fake_accuracy = T.lt(_evaluator(fake_images), swft.floatX(0.5)).mean()
    accuracy = (real_accuracy + fake_accuracy) / swft.floatX(2)

    real_train, real_dev, real_test = swft.mnist.load(BATCH_SIZE)

    assert(len(fakes) == 60000)
    fakes_train = fakes[:50000]
    fakes_dev   = fakes[50000:]

    def train_epoch():
        numpy.random.shuffle(fakes_train)
        batched = fakes_train.reshape(-1, BATCH_SIZE, 784)
        for i, (real_images, _) in enumerate(real_train()):
            yield [real_images, batched[i]]

    def dev_epoch():
        yield [real_dev().next()[0], fakes_dev]

    swft.train(
        [real_images, fake_images],
        [cost],
        train_epoch,
        dev_data=dev_epoch,
        epochs=EPOCHS,
        print_every=1000
    )

    fn = theano.function([real_images, fake_images], cost)
    result = fn(real_dev().next()[0], fakes_dev)

    swft.delete_params('Evaluator')

    return result