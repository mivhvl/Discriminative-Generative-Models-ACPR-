import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Define input image dimensions
IMG_SHAPE = (128, 128, 3)
LATENT_DIM = 100  # Size of the noise vector

def build_generator():
    model = keras.Sequential([
        layers.Dense(16 * 16 * 64, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((16, 16, 64)),
        
        layers.Conv2DTranspose(16, (5, 5), strides=(4, 4), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model


def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=IMG_SHAPE),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gan():
    generator = build_generator()
    discriminator = build_discriminator()

    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

    discriminator.trainable = False

    gan_input = keras.Input(shape=(LATENT_DIM,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)

    gan = keras.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return discriminator, generator, gan

def train_gan(dataset, discriminator, generator, gan, epochs=10000, batch_size=128):
    half_batch = batch_size // 2
   
    for epoch in range(epochs):
        d_losses = []
        g_losses = []
        batches = dataset.take(39)
        for batch in batches:
            real_images = batch
            noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
            fake_images = generator.predict(noise)

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_losses.append(d_loss[0])
            discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            g_losses.append(g_loss)
        print(f"Epoch {epoch}, D Loss: {np.mean(d_losses)}, G Loss: {np.mean(g_losses)}")
        generate_and_save_images(epoch, generator)

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
