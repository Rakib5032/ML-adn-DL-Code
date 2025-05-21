import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, train_ratio=0.80, val_ratio=0.0, test_ratio=0.20):
    """
    Splits a dataset into train, validation (optional), and test sets.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Input directory '{input_dir}' does not exist.")
        return

    if not (train_ratio + val_ratio + test_ratio == 1.0):
        print("Train, validation, and test ratios must sum to 1.0.")
        return

    # Prepare output directories
    splits = ['train', 'test'] if val_ratio == 0 else ['train', 'val', 'test']
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

    for category in os.listdir(input_path):
        category_path = input_path / category
        if category_path.is_dir():
            images = [f for f in category_path.iterdir() if f.suffix.lower() in image_extensions]

            train_images, remaining_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)

            if val_ratio == 0:
                test_images = remaining_images
                splits_to_apply = ['train', 'test']
                images_split = [train_images, test_images]
            else:
                val_images, test_images = train_test_split(
                    remaining_images,
                    test_size=(test_ratio / (val_ratio + test_ratio)),
                    random_state=42
                )
                splits_to_apply = ['train', 'val', 'test']
                images_split = [train_images, val_images, test_images]

            for split, split_images in zip(splits_to_apply, images_split):
                split_category_path = output_path / split / category
                split_category_path.mkdir(parents=True, exist_ok=True)

                for image in split_images:
                    shutil.copy(image, split_category_path / image.name)
                    print(f"Copied {image} -> {split_category_path / image.name}")

if __name__ == "__main__":
    input_dir = r'D:\MLLLLLL\Augmented TestData'
    output_dir = r'D:\MLLLLLL\split3'

    split_dataset(input_dir, output_dir)
