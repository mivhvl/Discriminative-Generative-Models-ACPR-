from PIL import Image
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


    # Process images
    for filename in os.listdir(input_dir):
        standardize_1_file(input_dir, output_dir, filename, TARGET_SIZE)

    print("Normalization complete. Images saved in", output_dir)

standardize_1_file('data/real', './', '808_1899-08-13_1955.jpg', (128, 128))