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

def base_stats(real_dir, fake_dir):
    # Analyze images in both folders
    real_images = analyze_images(real_dir)
    fake_images = analyze_images(fake_dir)
    # Print summaries
    summarize_images(real_images, "Real")
    summarize_images(fake_images, "Fake")

    del real_images
    del fake_images