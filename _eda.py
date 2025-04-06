import os
from PIL import Image
from collections import Counter

def analyze_images(directory):
    image_info = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            with Image.open(file_path) as img:
                image_info.append((img.format, img.mode, img.size))
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return image_info

def summarize_images(image_list, label):
    formats = Counter(img[0] for img in image_list)
    modes = Counter(img[1] for img in image_list)
    sizes = Counter(img[2] for img in image_list)

    print(f"\n{label} Images Summary:")
    print(f"Total images: {len(image_list)}")
    print(f"Formats: {formats}")
    print(f"Color Modes: {modes}")
    print(f"Top 5 Sizes: {sizes.most_common(5)}")

def base_stats(dir):
    images = analyze_images(dir)
    summarize_images(images, "Real")
    del images