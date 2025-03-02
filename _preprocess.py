from PIL import Image
import os

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

    # Define target size and format
    # TARGET_SIZE = (512, 512)

    # Process images
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
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

    print("Normalization complete. Images saved in '"+output_dir+"'.")
