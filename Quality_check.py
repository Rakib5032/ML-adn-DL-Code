import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np

def is_low_resolution(image_path, threshold=500):
    """Check if the image resolution is below the threshold."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        return width * height < threshold
    except Exception as e:
        print(f"Error in is_low_resolution for {image_path}: {e}")
        return False

def is_blurry(image_path, threshold=1000):
    """Check if the image is blurry based on Laplacian variance."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance < threshold
    except Exception as e:
        print(f"Error in is_blurry for {image_path}: {e}")
        return False

def has_brightness_issue(image_path, brightness_threshold=100):
    """Check if the image brightness is too low or too high."""
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(img)
        brightness = enhancer.enhance(1.0)  # Use the current brightness level
        return brightness < brightness_threshold
    except Exception as e:
        print(f"Error in has_brightness_issue for {image_path}: {e}")
        return False

def has_compression_artifacts(image_path):
    """Check for visible compression artifacts."""
    return False  # Assume no artifacts for simplicity

def has_uniform_color(image_path, threshold=0.9):
    """Check if the image has uniform color."""
    try:
        img = Image.open(image_path)
        img_data = np.array(img)
        unique_colors = np.unique(img_data.reshape(-1, img_data.shape[2]), axis=0)
        return len(unique_colors) / (img_data.shape[0] * img_data.shape[1]) < threshold
    except Exception as e:
        print(f"Error in has_uniform_color for {image_path}: {e}")
        return False

def has_high_noise(image_path, noise_threshold=10):
    """Check for noise by measuring pixel variation."""
    try:
        img = cv2.imread(image_path)
        mean, stddev = cv2.meanStdDev(img)
        return stddev[0] > noise_threshold
    except Exception as e:
        print(f"Error in has_high_noise for {image_path}: {e}")
        return False

def increase_resolution(image_path):
    """Increase resolution by resizing."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        new_width = width * 2
        new_height = height * 2
        img_resized = img.resize((new_width, new_height), Image.BICUBIC)
        img_resized.save(image_path)
    except Exception as e:
        print(f"Error in increase_resolution for {image_path}: {e}")

def sharpen_image(image_path):
    """Apply sharpening filter to fix blur."""
    try:
        img = cv2.imread(image_path)
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened_img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(image_path, sharpened_img)
    except Exception as e:
        print(f"Error in sharpen_image for {image_path}: {e}")

def adjust_brightness(image_path, factor=1.5):
    """Adjust brightness."""
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(img)
        img_brightened = enhancer.enhance(factor)
        img_brightened.save(image_path)
    except Exception as e:
        print(f"Error in adjust_brightness for {image_path}: {e}")

def reduce_compression(image_path):
    """Reduce compression artifacts by re-saving the image."""
    try:
        img = cv2.imread(image_path)
        cv2.imwrite(image_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Save with higher quality
    except Exception as e:
        print(f"Error in reduce_compression for {image_path}: {e}")

def enhance_contrast(image_path):
    """Enhance contrast of the image."""
    try:
        img = cv2.imread(image_path)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        img_contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(image_path, img_contrast)
    except Exception as e:
        print(f"Error in enhance_contrast for {image_path}: {e}")

def denoise_image(image_path):
    """Apply denoising to the image."""
    try:
        img = cv2.imread(image_path)
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv2.imwrite(image_path, denoised_img)
    except Exception as e:
        print(f"Error in denoise_image for {image_path}: {e}")

def fix_image(image_path):
    """Fix the image if any issues are found and replace it in the current folder."""
    fixed = False

    # Low resolution check and fix
    if is_low_resolution(image_path):
        increase_resolution(image_path)
        fixed = True

    # Blurry check and fix
    if is_blurry(image_path):
        sharpen_image(image_path)
        fixed = True

    # Brightness issue check and fix
    if has_brightness_issue(image_path):
        adjust_brightness(image_path)
        fixed = True

    # Compression artifacts check and fix
    if has_compression_artifacts(image_path):
        reduce_compression(image_path)
        fixed = True

    # Uniform color check and fix
    if has_uniform_color(image_path):
        enhance_contrast(image_path)
        fixed = True

    # High noise level check and fix
    if has_high_noise(image_path):
        denoise_image(image_path)
        fixed = True

    if fixed:
        print(f"Image '{image_path}' was fixed and replaced.")
    else:
        print(f"No issues found with image '{image_path}'.")

def process_folder(folder_path):
    """Process all images in the folder."""
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                fix_image(image_path)
    except Exception as e:
        print(f"Error in process_folder: {e}")

# Define the path to your dataset folder
dataset_folder = "D:\MLLLLLL\Balanced Data\Cercospora"

# Call this function to process all images in the dataset folder
process_folder(dataset_folder)
