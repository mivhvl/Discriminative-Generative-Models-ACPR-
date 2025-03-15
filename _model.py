import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Define input image dimensions
IMG_SHAPE = (64, 64, 3)
LATENT_DIM = 100  # Size of the noise vector

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def build_generator():
    model = keras.Sequential([
        layers.Dense(8 * 8 * 128, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 128)),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),


        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.LeakyReLU(alpha=0.2),
        
        # State size: (64) x 32 x 32
        layers.Conv2D(64 * 2, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # State size: (64*2) x 16 x 16
        layers.Conv2D(64 * 4, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # State size: (64*4) x 8 x 8
        layers.Conv2D(64 * 8, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # State size: (64*8) x 4 x 4
        layers.Conv2D(1, kernel_size=4, strides=1, padding='valid', use_bias=False),
        layers.Activation('sigmoid')
    ])
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def graph_losses(d_loss, g_loss):
    plt.figure(figsize=(16, 8), dpi=150) 
  
    plt.plot(d_loss, label='Discriminator Loss', color='orange')

    plt.plot(g_loss, label='Generator Loss', color='blue')

    # adding title to the plot 
    plt.title('Loss per iteration (batch)') 
    
    # adding Label to the x-axis 
    plt.xlabel('iteration')
    plt.show()

class GAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, images, batch_size=128):
        noise = tf.random.normal([batch_size, LATENT_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_loss, gen_loss

def train_gan(gan, dataset, epochs=100, batch_size=128, debug_loss=True):
    e_d_losses = []
    e_g_losses = []

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for batch in dataset:
            d_loss, g_loss = gan.train_step(batch, batch_size=batch_size // 2)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
            e_d_losses.append(d_loss.numpy())
            e_g_losses.append(g_loss.numpy())

        display.clear_output(wait=True)
        e_d_loss = np.mean(d_losses)
        e_g_loss = np.mean(g_losses)
        print(f"Epoch {epoch}, D Loss: {e_d_loss}, G Loss: {e_g_loss}")

        if debug_loss:
            graph_losses(e_d_losses, e_g_losses)  # Ensure you have this function defined
        generate_and_save_images(epoch, gan.generator)  # Ensure this function is also defined

    return e_d_losses, e_g_losses

def build_gan(g_lr=1e-4, d_lr=1e-4):
    generator = build_generator()
    discriminator = build_discriminator()

    g_optimizer = keras.optimizers.Adam(g_lr)
    d_optimizer = keras.optimizers.Adam(d_lr)

    gan = GAN(generator, discriminator)
    gan.compile(g_optimizer, d_optimizer)
    
    return gan


def generate_and_save_images(epoch, generator):
    noise = np.random.normal(0, 1, (16, LATENT_DIM))
    gen_images = generator.predict(noise)
    gen_images = (gen_images + 1) / 2  # Rescale images to [0,1]
    
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_images[i])
        ax.axis('off')
    plt.savefig(f'gan_images/epoch_{epoch}.png')
    plt.close()

