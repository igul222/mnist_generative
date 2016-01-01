# MNIST Generative Models

Theano implementations of two generative models for MNIST:

- An adversarial autoencoder (Makhzani et al.)
- A generative adversarial network (Goodfellow et al.)

Both perform reasonably, but not particularly well. To run the code, you'll need to install Lasagne, as well as [this small Theano library](https://github.com/igul222/swft) I wrote.

Samples from the adversarial autoencoder:

![Adversarial autoencoder samples](https://raw.github.com/igul222/mnist_generative/blob/master/autoencoder.png)

Samples from the GAN:

![GAN samples](https://raw.github.com/igul222/mnist_generative/blob/master/gan.png)