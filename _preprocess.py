from PIL import Image, ImageOps
import os

def standardize_1_file(input_dir, output_dir, filename, TARGET_SIZE):
    file_path = os.path.join(input_dir, filename)
    print(file_path)
    try:
        with Image.open(file_path) as img:
            # Convert to RGB
            img = img.convert("RGB")
            # Resize
            img = img.resize(TARGET_SIZE)
            # Save as JPEG
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            img.save(output_path, "JPEG")
    except Exception as e:
        print(f"Skipping {filename}: {e}")

def standardize_files(input_dir, TARGET_SIZE, suffix=''):

    output_dir = input_dir + "_normalized_" + suffix
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define target size and format
    # TARGET_SIZE = (512, 512)

    # Process images
    for filename in os.listdir(input_dir):
        standardize_1_file(input_dir, output_dir, filename, TARGET_SIZE)
        

    print("Normalization complete. Images saved in 'data/real_normalized'.")

def augment_1_file(input_dir, output_dir, filename):
    file_path = os.path.join(input_dir, filename)
    print(file_path)
    try:
        basePath = os.path.join(output_dir, filename)
        with Image.open(file_path) as img:
            img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_flipped.save(basePath + '.flipped.jpg', "JPEG")

            img_mirrored = ImageOps.mirror(img)
            img_mirrored.save(basePath + '.mirrored.jpg', "JPEG")

            img_rotate_r = img.rotate(45)
            img_rotate_r.save(basePath + '.rotate.jpg', "JPEG")

            img_rotate_l = img.rotate(-45)
            img_rotate_l.save(basePath + '.rotate_l.jpg', "JPEG")

            img_contrast = ImageOps.autocontrast(img)
            img_contrast.save(basePath + '.contrast.jpg', "JPEG")

            img.save(basePath + '.jpg', "JPEG")


    except Exception as e:
        print(f"Skipping {filename}: {e}")

def augment_files(input_dir):
    output_dir = input_dir + "_augmented"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        augment_1_file(input_dir, output_dir, filename)
