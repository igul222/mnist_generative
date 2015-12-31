"""Adversarial Autoencoder for MNIST."""

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import swft
import lasagne

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE   = 100
INPUT_DIM    = 784
HIDDEN_DIM   = 1024
LATENT_DIM   = 8
LATENT_STDEV = 10

def Layer(name, n_in, n_out, batchnorm, inputs):
    """A ReLU layer, optionally with batch norm."""
    output = swft.ops.Linear(name+'.Linear', n_in, n_out, inputs)
    if batchnorm:
        output = swft.ops.BatchNormalize(name+'.BN', n_out, output)
    return swft.ops.rectify(output)

def encoder(images):
    output = Layer('Encoder.Layer1', INPUT_DIM,  HIDDEN_DIM, True, images)
    output = Layer('Encoder.Layer2', HIDDEN_DIM, HIDDEN_DIM, True, output)
    return   swft.ops.Linear('Encoder.Layer3', HIDDEN_DIM, LATENT_DIM, output)

def decoder(latent_vars):
    output = Layer('Decoder.Layer1', LATENT_DIM, HIDDEN_DIM, True, latent_vars)
    output = Layer('Decoder.Layer2', HIDDEN_DIM, HIDDEN_DIM, True, output)
    return   T.nnet.sigmoid(
        swft.ops.Linear('Decoder.Layer3', HIDDEN_DIM, INPUT_DIM, output)
    )

def discriminator(inputs):
    output = Layer('Discriminator.Layer1', LATENT_DIM, HIDDEN_DIM, False, inputs)
    output = Layer('Discriminator.Layer2', HIDDEN_DIM, HIDDEN_DIM, True,  output)
    return T.nnet.sigmoid(
        swft.ops.Linear('Discriminator.Layer3', HIDDEN_DIM, 1, output).flatten()
    )

theano_srng = RandomStreams(seed=234)
def noise(n_samples):
    output = theano_srng.normal(size=(n_samples,LATENT_DIM))
    return swft.floatX(LATENT_STDEV) * output

images, targets = swft.mnist.symbolic_inputs()

latents = encoder(images)
reconstructions = decoder(latents)

# Encoder objective:  push D(latents) to one...
reg_cost = T.nnet.binary_crossentropy(discriminator(latents), swft.floatX(1)).mean()
reg_cost.name = 'reg_cost'

# ... and minimize reconstruction error
reconst_cost = T.sqr(reconstructions - images).mean()
reconst_cost.name = 'reconst_cost'

# this seems to be an important hyperparam, maybe try playing with it more.
full_enc_cost = (swft.floatX(100)*reconst_cost) + reg_cost

# Decoder objective: minimize reconstruction loss
dec_cost = reconst_cost

# Discrim objective: push D(latents) to zero, D(noise) to one
discrim_cost  = T.nnet.binary_crossentropy(discriminator(latents),           swft.floatX(0)).mean()
discrim_cost += T.nnet.binary_crossentropy(discriminator(noise(BATCH_SIZE)), swft.floatX(1)).mean()
discrim_cost.name = 'discrim_cost'

enc_params     = swft.search(full_enc_cost, lambda x: hasattr(x, 'param') and 'Encoder' in x.name)
dec_params     = swft.search(dec_cost,      lambda x: hasattr(x, 'param') and 'Decoder' in x.name)
discrim_params = swft.search(discrim_cost,  lambda x: hasattr(x, 'param') and 'Discriminator' in x.name)

# Load dataset
train_data, dev_data, test_data = swft.mnist.load(BATCH_SIZE)

# sample_fn is used by generate_images
sample_fn = theano.function(
    [images], 
    [decoder(noise(100)), decoder(encoder(images[:100])), encoder(images)]
)
def generate_images(epoch):
    """
    Save samples and diagnostic images from the model. This function is passed
    as a callback to `train` and is called after every epoch.
    """
    def save_images(images, filename):
        images = images.reshape((10,10,28,28))
        # rowx, rowy, height, width -> rowy, height, rowx, width
        images = images.transpose(1,2,0,3)
        images = images.reshape((10*28, 10*28))
        plt.clf()
        plt.cla()
        plt.imshow(images, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.savefig(filename+'_epoch'+str(epoch))

    images, targets = dev_data().next()
    samples, reconstructions, latents = sample_fn(images)

    save_images(samples, 'samples')
    save_images(reconstructions, 'reconstructions')

    # Save a scatterplot of the first two dims of the latent representation    
    plt.clf()
    plt.cla()
    plt.scatter(*(latents[:,0:2].T), c=targets)
    plt.xlim(-4*LATENT_STDEV, 4*LATENT_STDEV)
    plt.ylim(-4*LATENT_STDEV, 4*LATENT_STDEV)
    plt.savefig('latents_epoch'+str(epoch))

# Start training!
swft.train(
    [images, targets],
    [full_enc_cost, dec_cost, discrim_cost],
    param_sets  = [enc_params, dec_params, discrim_params],
    optimizers  = [
        lasagne.updates.adam,
        lasagne.updates.adam,
        lasagne.updates.adam
    ],
    print_vars  = [reg_cost, reconst_cost, discrim_cost],
    train_data  = train_data,
    dev_data    = dev_data,
    epochs      = 100,
    callback    = generate_images,
    print_every = 1000
)