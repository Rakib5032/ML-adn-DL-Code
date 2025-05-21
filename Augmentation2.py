import os
import uuid
import logging
from tqdm import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
input_dir = r'D:\MLLLLLL\Test Dataset\Cercospora'
output_dir = r'D:\MLLLLLL\Augmented TestData'
padding_size = 50
num_augmentations = 2
target_size = (624, 624)
seed = 42  # Valid only for .flow()

def add_padding(img, padding_size=50, color=(255, 255, 255)):
    """
    Add padding around the image.
    Args:
    - img: PIL Image object
    - padding_size: Size of padding in pixels
    - color: Padding color
    """
    width, height = img.size
    new_width = width + 2 * padding_size
    new_height = height + 2 * padding_size
    new_img = Image.new("RGB", (new_width, new_height), color)
    new_img.paste(img, (padding_size, padding_size))
    return new_img

def setup_datagen():
    """
    Set up the ImageDataGenerator with augmentations.
    """
    return ImageDataGenerator(
        rotation_range=50,
        shear_range=0.1,
        zoom_range=0.,
        horizontal_flip=True,
        fill_mode='constant',
        cval=255  # Fill value for constant mode
    )

def augment_images(input_dir, output_dir, padding_size, num_augmentations):
    """
    Perform image augmentation on all images in the given directory.
    Args:
    - input_dir: Directory containing the original images
    - output_dir: Directory where augmented images will be saved
    - padding_size: Padding size around the images
    - num_augmentations: Number of augmentations to generate per image
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory {input_dir} does not exist.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    class_name = os.path.basename(input_dir)
    class_output_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(class_output_dir):
        os.makedirs(class_output_dir)
        logging.info(f"Created class directory: {class_output_dir}")

    datagen = setup_datagen()

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))]
    if not image_files:
        logging.warning(f"No valid images found in {input_dir}.")
        return

    total_augmented = 0

    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            padded_img = add_padding(img, padding_size)
            # Resize without normalizing pixel values
            padded_img = padded_img.resize(target_size)

            x = tf.keras.preprocessing.image.img_to_array(padded_img)  # Keeps original pixel values
            x = np.expand_dims(x, axis=0)

            # Generate augmentations
            i = 0
            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=class_output_dir,
                save_prefix=f"aug_{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:5]}",
                save_format='jpeg',
                seed=seed  # Ensure reproducibility of augmentations
            ):
                i += 1
                total_augmented += 1
                if i >= num_augmentations:
                    break

        except Exception as e:
            logging.error(f"Failed to process {filename}: {str(e)}")
            continue

    logging.info(f"Augmentation complete: {total_augmented} images saved to {class_output_dir}")

def main():
    augment_images(input_dir, output_dir, padding_size, num_augmentations)

if __name__ == "__main__":
    main()
