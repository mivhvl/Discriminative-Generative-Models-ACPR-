import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

# Get random images for the augmentation
def get_random_filenames(filenames, num_images):
    random.shuffle(filenames)  # Shuffle filenames to ensure randomness
    return filenames[:num_images]

# Flip horizontally and adjust hue
def flip_hue_augmentation(input_folder, output_folder, filenames, num_augmented_images=5):
    augmented_count = 0
    random_filenames = get_random_filenames(filenames, num_augmented_images)

    for filename in random_filenames:
        img_path = os.path.join(input_folder, filename)

        if os.path.isfile(img_path):
            img = image.load_img(img_path)
            img_array = image.img_to_array(img)

            flipped_image = tf.image.flip_left_right(img_array)
            hue_adjusted_image = tf.image.adjust_hue(flipped_image, delta=0.1)
            hue_adjusted_image = np.array(hue_adjusted_image, dtype=np.uint8)
            hue_pil_image = Image.fromarray(hue_adjusted_image)

            new_filename = f'{os.path.splitext(filename)[0]}_fh.jpg'
            hue_pil_image.save(os.path.join(output_folder, new_filename))

            augmented_count += 1

    print(f"Horizontal flip and hue adjustment complete, {augmented_count} images saved.")

# Change brightness and hue
def brightness_hue_augmentation(input_folder, output_folder, filenames, num_augmented_images=5, brightness_delta=0.2, hue_delta=0.05):
    augmented_count = 0
    random_filenames = get_random_filenames(filenames, num_augmented_images)

    for filename in random_filenames:
        img_path = os.path.join(input_folder, filename)

        if os.path.isfile(img_path):
            img = image.load_img(img_path)
            img_array = image.img_to_array(img)

            if len(img_array.shape) == 3 and img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)

            brightened_image = tf.image.adjust_brightness(img_array, delta=brightness_delta)
            hue_adjusted_image = tf.image.adjust_hue(brightened_image, delta=hue_delta)

            hue_brightness_adjusted_image = np.array(hue_adjusted_image, dtype=np.uint8)
            hue_brightness_pil_image = Image.fromarray(hue_brightness_adjusted_image)

            new_filename = f'{os.path.splitext(filename)[0]}_bh.jpg'
            hue_brightness_pil_image.save(os.path.join(output_folder, new_filename))

            augmented_count += 1

    print(f"Brightness and hue adjustment complete, {augmented_count} images saved.")

# Zoom the image
def zoom_augmentation(input_folder, output_folder, filenames, num_augmented_images=5, zoom_factor=0.2):
    augmented_count = 0
    random_filenames = get_random_filenames(filenames, num_augmented_images)

    for filename in random_filenames:
        img_path = os.path.join(input_folder, filename)

        if os.path.isfile(img_path):
            img = image.load_img(img_path)
            img_array = image.img_to_array(img)

            height, width, _ = img_array.shape
            zoomed_image = tf.image.resize(img_array, [int(height * (1 + zoom_factor)), int(width * (1 + zoom_factor))])
            cropped_image = tf.image.resize_with_crop_or_pad(zoomed_image, target_height=height, target_width=width)
            cropped_image = np.array(cropped_image, dtype=np.uint8)
            zoomed_pil_image = Image.fromarray(cropped_image)

            # Add 'd' suffix to avoid overwriting
            new_filename = f'{os.path.splitext(filename)[0]}_z.jpg'
            zoomed_pil_image.save(os.path.join(output_folder, new_filename))

            augmented_count += 1

    print(f"Zoom augmentation complete, {augmented_count} images saved.")
