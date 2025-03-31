import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from _fid import *

# Define input image dimensions
IMG_SHAPE = (64, 64, 3)
LATENT_DIM = 100  # Size of the noise vector

cross_entropy = tf.keras.losses.BinaryCrossentropy()


def build_generator(leak=False):
    if leak:
        act = layers.LeakyReLU
    else:
        act = layers.ReLU
    model = keras.Sequential([
        layers.Dense(8 * 8 * 128, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        act(),
        layers.Reshape((8, 8, 128)),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        act(),


        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        act(),
        
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

def discriminator_loss(real_output, fake_output, label_smoothing=False):
    if label_smoothing:
        real_loss = cross_entropy(tf.ones_like(real_output, dtype=tf.float32)*0.9, real_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def graph_losses(d_loss=None, g_loss=None):
    plt.figure(figsize=(16, 8), dpi=150) 

    if d_loss is not None:
        plt.plot(d_loss, label='Discriminator Loss', color='orange')

    if g_loss is not None:
        plt.plot(g_loss, label='Generator Loss', color='blue')

    # adding title to the plot 
    plt.title('Loss per iteration (batch)') 
    
    # adding Label to the x-axis 
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.show()

class GAN(keras.Model):
    def __init__(self, generator, discriminator, label_smoothing, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.label_smoothing = label_smoothing

    def get_config(self):
        config = super(GAN, self).get_config()
        config.update({
            "label_smoothing": self.label_smoothing,
            "generator_json": self.generator.to_json(),
            "discriminator_json": self.discriminator.to_json()
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.models.model_from_json(config.pop("generator_json"))
        discriminator = keras.models.model_from_json(config.pop("discriminator_json"))
        return cls(generator=generator, discriminator=discriminator, **config)

    def compile(self, g_optimizer, d_optimizer):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def get_compile_config(self):
        return {
            "g_optimizer": keras.optimizers.serialize(self.g_optimizer),
            "d_optimizer": keras.optimizers.serialize(self.d_optimizer)
        }

    def compile_from_config(self, config):
        """Restores the optimizer configuration."""
        self.compile(
            g_optimizer=keras.optimizers.deserialize(config["g_optimizer"]),
            d_optimizer=keras.optimizers.deserialize(config["d_optimizer"])
        )

    def train_step(self, images, batch_size=128):
        noise = tf.random.normal([batch_size, LATENT_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output, self.label_smoothing)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_loss, gen_loss

def train_gan(gan, dataset, epochs=100, batch_size=128, debug_loss=True, fid=True):
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

        if fid and epoch % 50 == 0:
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
            gen_images = gan.generator.predict(noise)
            real_images = dataset.take(1)
            real_images = next(iter(dataset.take(1)))[0].numpy()
            # Ensure correct dtype for skimage
            real_images = real_images.astype(np.float32)
            gen_images = img_scaler(gen_images, (75,75,3))
            real_images = img_scaler(real_images, (75,75,3))
            fid_score = calculate_fid(inception_model, gen_images, real_images)
            print('FID SCORE: ', fid_score)
        if debug_loss:
            graph_losses(e_d_losses, e_g_losses)  # Ensure you have this function defined
        generate_and_save_images(epoch, gan.generator)  # Ensure this function is also defined

    return e_d_losses, e_g_losses

def build_gan(generator=None, discriminator=None, g_lr=1e-4, d_lr=1e-4, label_smoothing=False):
    if generator is None:
        generator = build_generator()
    
    if discriminator is None:
        discriminator = build_discriminator()

    g_optimizer = keras.optimizers.Adam(g_lr)
    d_optimizer = keras.optimizers.Adam(d_lr)

    gan = GAN(generator, discriminator, label_smoothing)
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

def train_discriminator(discriminator, dataset_real, dataset_fake, epochs=100, debug_loss=True, batch_size=128, label_smoothing=False):
    e_d_total_losses = []
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) # You'll need an optimizer

    # Create iterators for the datasets to manually fetch batches
    real_iterator = iter(dataset_real)
    fake_iterator = iter(dataset_fake)

    num_batches = len(list(dataset_real.as_numpy_iterator())) # Get the number of batches in the real dataset

    for epoch in range(epochs):
        d_losses = []
        try:
            for _ in range(num_batches):
                batch_real = next(real_iterator)
                batch_fake = next(fake_iterator) # Ensure fake batch is available

                # Ensure both batches have the same size or handle potential mismatch
                current_batch_size = batch_real.shape[0]
                if batch_fake.shape[0] < current_batch_size:
                    # Pad or truncate batch_fake to match batch_real size
                    padding_needed = current_batch_size - batch_fake.shape[0]
                    padding = np.zeros((padding_needed,) + batch_fake.shape[1:], dtype=batch_fake.dtype)
                    batch_fake = np.concatenate([batch_fake, padding], axis=0)
                elif batch_fake.shape[0] > current_batch_size:
                    batch_fake = batch_fake[:current_batch_size]

                d_loss = train_discriminator_step(discriminator, batch_real, batch_fake, discriminator_optimizer, label_smoothing=label_smoothing)
                d_losses.append(d_loss.numpy())
                e_d_total_losses.append(d_loss.numpy())

        except StopIteration:
            print("Datasets exhausted. Consider resetting iterators if needed for more epochs.")
            break

        display.clear_output(wait=True)
        e_d_loss = np.mean(d_losses)
        print(f"Epoch {epoch+1}, D Loss: {e_d_loss}")

    if debug_loss:
        graph_losses(e_d_total_losses, None)


# Define the training step for the discriminator
@tf.function
def train_discriminator_step(discriminator, real_images, fake_images, discriminator_optimizer, label_smoothing=False):
    with tf.GradientTape() as disc_tape:
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        disc_loss = discriminator_loss(real_output, fake_output, label_smoothing=label_smoothing)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return disc_loss

