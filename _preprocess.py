from PIL import Image, ImageOps
import os
import random
import shutil
from sklearn.model_selection import train_test_split
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def standardize_1_file(input_dir, output_dir, filename, TARGET_SIZE=False):
    file_path = os.path.join(input_dir, filename)
    print(file_path)
    try:
        with Image.open(file_path) as img:
            # Convert to RGB
            img = img.convert("RGB")
            # Resize
            if TARGET_SIZE:
                img = img.resize(TARGET_SIZE)
            # Save as JPEG
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            img.save(output_path, "JPEG")
    except Exception as e:
        print(f"Skipping {filename}: {e}")

def standardize_files(input_dir, TARGET_SIZE=False, suffix=''):

    output_dir = input_dir + "_normalized" + suffix
def standardize_files(input_dir, TARGET_SIZE, output_dir):
    """
    Standardizes images in a given directory by resizing and converting them to JPEG format.

    Parameters:
    input_dir (str): Path to the directory containing the input images.
    TARGET_SIZE (tuple): The target size for resizing images, specified as (width, height).
    output_dir (str or None): Path to the directory where standardized images will be saved. 
                              If None, a new directory with suffix '_normalized' will be created.

    The function resizes all images in the input directory to the specified target size, converts 
    them to RGB color mode, and saves them as JPEG files in the output directory. If an image 
    cannot be processed, it will be skipped with an error message.
    """
    
    if output_dir is None:
        output_dir = input_dir + "_normalized"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    # Process images
    for filename in os.listdir(input_dir):
        standardize_1_file(input_dir, output_dir, filename, TARGET_SIZE)

    print("Normalization complete. Images saved in", output_dir)

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

def select_and_remove_images(source_dir, num_images_to_keep=20000):
    # Take images paths
    all_images = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if len(all_images) <= num_images_to_keep:
        print(f"Not enough images to remove. Keeping {len(all_images)} images.")
        return

    # Shuffle and remove random images
    random.shuffle(all_images)
    images_to_remove = all_images[num_images_to_keep:]
    for image in images_to_remove:
        os.remove(image)

    print(f"Removed {len(images_to_remove)} images.")

def create_train_val_test_split(real_dirs, fake_dirs, test_val_size_real=0.15, test_val_size_fake=0.15):
    # Collect real images paths
    real_images = []
    for real_dir in real_dirs:
        for real_dir_item in os.listdir(real_dir):
            real_images.append(os.path.join(real_dir, real_dir_item))

    # Collect fake images paths
    fake_images = []
    for fake_dir in fake_dirs:
        for fake_dir_item in os.listdir(fake_dir):
            fake_images.append(os.path.join(fake_dir, fake_dir_item))

    # Shuffle real and fake images
    random.shuffle(real_images)
    random.shuffle(fake_images)

    #Perform the train/test split
    real_train, real_temp = train_test_split(real_images, test_size=test_val_size_real * 2, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
    fake_train, fake_temp = train_test_split(fake_images, test_size=test_val_size_fake * 2, random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=0.5, random_state=42)

    # Display Dataset split
    print('Real Train: ', len(real_train))
    print('Real Val: ', len(real_val))
    print('Real Test: ', len(real_test))
    print('Fake Train: ', len(fake_train))
    print('Fake Val: ', len(fake_val))
    print('Fake Test: ', len(fake_test))

    return real_train, real_val, real_test, fake_train, fake_val, fake_test


# Copy images with a unique names
def copy_image_with_unique_name(src, dest_dir):
    file_name = os.path.basename(src)
    dest_path = os.path.join(dest_dir, file_name)
    base_name, ext = os.path.splitext(file_name)
    counter = 1

    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    shutil.copy(src, dest_path)

# Copy images to the correct directories (train, val, test)
def copy_images_to_dirs(images, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for img in images:
        copy_image_with_unique_name(img, dest_dir)

# Copy split data (train, val, test for real and fake images)
def copy_split_data(real_train, real_val, real_test, fake_train, fake_val, fake_test, output_dir):
    # Dictionaries structure
    os.makedirs(os.path.join(output_dir, 'train/real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train/fake'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val/real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val/fake'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test/real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test/fake'), exist_ok=True)

    # Copy images to the directories
    copy_images_to_dirs(real_train, os.path.join(output_dir, 'train/real'))
    copy_images_to_dirs(real_val, os.path.join(output_dir, 'val/real'))
    copy_images_to_dirs(real_test, os.path.join(output_dir, 'test/real'))
    copy_images_to_dirs(fake_train, os.path.join(output_dir, 'train/fake'))
    copy_images_to_dirs(fake_val, os.path.join(output_dir, 'val/fake'))
    copy_images_to_dirs(fake_test, os.path.join(output_dir, 'test/fake'))


def detect_and_crop_faces(input_root, output_folder, target_size=(100, 100), model_path='yolov8n-face-lindevs.pt'):

    # Load YOLOv8 model
    model = YOLO(model_path)
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Process all images recursively
    for root, _, files in os.walk(input_root):
        for filename in tqdm(files, desc=f"Processing {root}", disable=True):
            img_path = os.path.join(root, filename)
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip non-image files
            # Run YOLO detection
            results = model(img)
            # If no face detected, skip
            if len(results[0].boxes) == 0:
                continue
            # Extract first detected face
            x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue
            # Convert to PIL format
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            # Resize cropped face to the target size
            pil_img_resized = pil_img.resize(target_size)
            # Generate a unique filename
            new_filename = f"{os.path.splitext(filename)[0]}_cropped.jpg"
            save_path = os.path.join(output_folder, new_filename)
            # Save resized cropped face
            pil_img_resized.save(save_path)

    print(f"Processing complete! Cropped and resized images saved in '{output_folder}'.")

