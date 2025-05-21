from rembg import remove
from PIL import Image
import os
import io

def remove_bg_and_make_white_recursive(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)

                # Get relative path to maintain folder structure
                rel_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, rel_path)
                output_path = os.path.join(output_dir, file)

                # Create the output subdirectory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)

                try:
                    # Read and remove background
                    with open(input_path, "rb") as input_file:
                        input_image = input_file.read()
                        output_image = remove(input_image)

                    # Replace transparency with white
                    with Image.open(io.BytesIO(output_image)) as img:
                        if img.mode in ('RGBA', 'LA'):
                            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                            img = Image.alpha_composite(background, img).convert("RGB")

                        img.save(output_path)
                        print(f"Processed and saved: {output_path}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# Example usage
input_folder = r"D:\ML Bishal\Origanal Data 1"
output_folder = r"D:\ML Bishal\BGremoved"

remove_bg_and_make_white_recursive(input_folder, output_folder)
