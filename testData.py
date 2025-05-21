import cv2
import numpy as np
from rembg import remove
from PIL import Image, UnidentifiedImageError
import io
import os

def segment_leaf_by_color(image):
    """Segment the leaf using color-based segmentation in HSV space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_leaf = np.array([30, 50, 50])
    upper_leaf = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    return leaf_mask

def detect_edges_and_create_mask(leaf_mask):
    """Detect edges and enhance the mask."""
    edges = cv2.Canny(leaf_mask, threshold1=100, threshold2=200)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    inverted_edges = cv2.bitwise_not(edges_dilated)
    combined_mask = cv2.bitwise_and(leaf_mask, inverted_edges)
    return combined_mask

def process_and_save_image(image_path, output_path):
    """Process a single image by combining color segmentation, edge detection, and rembg background removal."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

    # Apply color-based segmentation and edge detection
    leaf_mask = segment_leaf_by_color(image)
    edges = detect_edges_and_create_mask(leaf_mask)

    # Remove background using rembg
    _, buffer = cv2.imencode('.png', image)
    removed_bg = remove(buffer.tobytes(), force_return_bytes=True)
    result_image = Image.open(io.BytesIO(removed_bg)).convert('RGBA')
    
    # Create a white background
    white_bg = Image.new('RGBA', result_image.size, (255, 255, 255, 255))

    # Composite the image onto the white background
    result_image = Image.alpha_composite(white_bg, result_image).convert('RGB')

    # Apply mask to enhance segmentation
    refined_image = np.array(result_image)
    combined_mask = cv2.bitwise_and(refined_image, refined_image, mask=edges)

    # Convert to PIL and resize
    final_result = Image.fromarray(combined_mask)
    final_result = final_result.resize((640, 640), Image.LANCZOS)

    # Save final result with a white background
    final_result.save(output_path, format='JPEG', quality=100, optimize=True)
    print(f"Processed and saved: {output_path}")

def process_images_in_folder(input_folder, output_folder):
    """Process each image in the input folder and save it to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    for image_file in image_files:
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, f"processed_{image_file}")
        
        try:
            process_and_save_image(input_image_path, output_image_path)
        except UnidentifiedImageError:
            print(f"Error: {image_file} is not a valid image file. Skipping.")
        except Exception as e:
            print(f"Unexpected error processing {image_file}: {e}")

    print(f"Processed {len(image_files)} images. The results are saved in {output_folder}.")

# Define the input and output folder paths
input_folder = "D:\\Data Set\\Hadda Beetle"  # Replace with your input folder
output_folder = "D:\\TEST\\test"  # Replace with your output folder

# Process images in the input folder
process_images_in_folder(input_folder, output_folder)
