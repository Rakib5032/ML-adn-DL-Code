from rembg import remove
from PIL import Image
import os
import io

def remove_bg_and_make_white(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Read the image and remove the background
                with open(input_path, "rb") as input_file:
                    input_image = input_file.read()
                    output_image = remove(input_image)

                # Load the processed image and replace transparent background with white
                with Image.open(io.BytesIO(output_image)) as img:
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                        img = Image.alpha_composite(background, img).convert("RGB")

                    # Save the output image
                    img.save(output_path)
                    print(f"Processed and saved: {output_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# Set the input and output folder paths
input_folder = r"D:\ML Bishal\Augmented Data"
output_folder = r"D:\ML Bishal\BGremoved"

# Call the function
remove_bg_and_make_white(input_folder, output_folder)
