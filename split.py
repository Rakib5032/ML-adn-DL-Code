import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits a dataset into train, validation, and test sets.

    :param input_dir: Path to the folder containing category subfolders with images.
    :param output_dir: Path to the folder where train, val, and test folders will be created.
    :param train_ratio: Proportion of data to be used for training (default: 0.7).
    :param val_ratio: Proportion of data to be used for validation (default: 0.15).
    :param test_ratio: Proportion of data to be used for testing (default: 0.15).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Input directory '{input_dir}' does not exist.")
        return

    if not (train_ratio + val_ratio + test_ratio == 1.0):
        print("Train, validation, and test ratios must sum to 1.0.")
        return

    # Create output directories for train, val, and test
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        split_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

    # Process each subfolder (category)
    for category in os.listdir(input_path):
        category_path = input_path / category
        if category_path.is_dir():
            images = [f for f in category_path.iterdir() if f.suffix.lower() in image_extensions]

            # Split the images
            train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

            # Define destination paths
            for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
                split_category_path = output_path / split / category
                split_category_path.mkdir(parents=True, exist_ok=True)

                # Copy images to their respective folders
                for image in split_images:
                    shutil.copy(image, split_category_path / image.name)
                    print(f"Copied {image} -> {split_category_path / image.name}")

if __name__ == "__main__":
    # Input and output directory paths
    input_dir = 'D:\MLLLLLL\Augmented TestData'
    output_dir = 'D:\MLLLLLL\split3'

    split_dataset(input_dir, output_dir)
