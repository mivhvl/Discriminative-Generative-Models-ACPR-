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

def build_gan(g_lr=1e-4, d_lr=1e-4):
    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(g_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(d_lr)
    return discriminator, generator, generator_optimizer, discriminator_optimizer

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train_step(images, discriminator, generator, discriminator_optimizer, generator_optimizer, batch_size=128):
    noise = tf.random.normal([batch_size, LATENT_DIM])

    disc_loss = gen_loss = 0
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return disc_loss, gen_loss

def train_gan(dataset, discriminator, generator, discriminator_optimizer, generator_optimizer, epochs=100, batch_size=128):
    half_batch = batch_size // 2
    e_d_losses = []
    e_g_losses = []

    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        for batch in dataset:
            d_loss, g_loss = train_step(batch, discriminator, generator, discriminator_optimizer, generator_optimizer, batch_size=half_batch)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
        display.clear_output(wait=True)
        e_d_loss = np.mean(d_losses)
        e_g_loss = np.mean(g_losses)
        e_d_losses.append(e_d_loss)
        e_g_losses.append(e_g_loss)
        print(f"Epoch {epoch}, D Loss: {e_d_loss}, G Loss: {e_g_loss}")
        generate_and_save_images(epoch, generator)
    return e_d_losses, e_g_losses
# Function to generate and save images
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
