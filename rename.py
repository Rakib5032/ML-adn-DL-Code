import os

def rename_images_based_on_folder(base_dir):
    """
    Rename images in each subfolder of the base directory sequentially starting from 1,
    appending the folder name to each image (e.g., FolderName_1.jpg, FolderName_2.jpg, ...).
    
    Args:
        base_dir (str): Path to the base directory containing subfolders of images.
    """
    for root, _, files in os.walk(base_dir):
        # Get the folder name (the name of the subfolder)
        folder_name = os.path.basename(root)

        # Skip the base directory itself
        if root == base_dir:
            continue

        # Filter image files only (jpg, jpeg, png)
        image_files = [
            file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        # Sort files to ensure consistent order
        image_files.sort()

        # Rename each file with folder name and sequential numbering starting from 1
        for idx, filename in enumerate(image_files, start=1):
            old_path = os.path.join(root, filename)
            new_filename = f"{folder_name}_{idx}.jpg"  # New filename: FolderName_1.jpg, FolderName_2.jpg, ...
            new_path = os.path.join(root, new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Error renaming {old_path}: {e}")

if __name__ == "__main__":
    # Set the base directory (change this to your actual directory path)
    base_directory = r"D:\\Eggplant Dataset"  # Example directory
    rename_images_based_on_folder(base_directory)
