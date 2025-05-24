# Utility functions for bounding box processing

# Imports

from hmac import new
import io
import os
from pickletools import optimize
from turtle import width
from pdf2image import convert_from_path
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
from typing import List, Tuple

from IPython.display import display


########
# Function to convert a multi-page PDF into a single JPEG image
def convert_pdf_to_pil_image(input_directory: str, input_file: str, pages: List[int]=[], stack: bool=False, output_directory: str="", dpi: int=150, quality: int=80, suffix: str="") -> Image.Image:
    """
    Convert a multi-page PDF into a single JPEG image.
    Args:
        directory (str): The directory where the PDF file is located.
        input_file (str): The name of the PDF file (without extension).
        pages (List[int], optional): List of page numbers to include. If None, all pages are included.
    Returns:
        Image: The combined JPEG image.
    """
    # Check if the input file exists
    if not os.path.exists(os.path.join(f"{input_directory}/", f"{input_file}.pdf")):
        print(f"File {input_file}.pdf does not exist.")

    # Convert PDF to list of PIL Image objects
    images = convert_from_path(os.path.join(f"{input_directory}/", f"{input_file}.pdf"), dpi=dpi)
    if len(pages) >= 1:
        images = [images[k1] for k1 in pages if k1 < len(images)]
    # images = [image.resize((int(image.width/2), int(image.height/2))) for image in images]
    images = [image.convert("L") for image in images]

    if stack:

        # Calculate total height and max width
        total_height = sum(image.height for image in images)
        max_width = max(image.width for image in images)

        # Create a new blank image with combined size
        combined_image = Image.new("RGB", (max_width, total_height))
        combined_image = combined_image.convert("L")
        
        # Paste all images vertically
        y_offset = 0
        for image in images:
            combined_image.paste(image, (0, y_offset))
            y_offset += image.height

        # Save as JPEG
        if output_directory == "" or os.path.exists(output_directory) == False:
            output_directory = input_directory
        combined_image.save(os.path.join(f"{output_directory}/", f"{input_file}_combined{suffix}.jpg"), "JPEG", quality=quality, optimize=True)

    else:

        # Save images and individual JPEGs
        if output_directory == "" or os.path.exists(output_directory) == False:
            output_directory = input_directory
        for k1, image in enumerate(images):
            image.save(os.path.join(f"{output_directory}/", f"{input_file}_{k1:02d}{suffix}.jpg"), "JPEG", quality=quality, optimize=True)
        
        


########
def resize_jpeg_image_ratio(directory: str, input_file: str, ratio: float=1.0, quality: int=100, suffix: str="") -> Image.Image:
    """
    Resize a JPEG image to half its original size.
    Args:
        directory (str): The directory where the JPEG file is located.
        input_file (str): The name of the JPEG file (without extension).
    Returns:
        Image: The resized JPEG image.
    """
    # Check if the input file exists
    if not os.path.exists(os.path.join(f"{directory}/", f"{input_file}.jpg")):
        print(f"File {input_file}.jpg does not exist.")

    # Load the image
    image = Image.open(os.path.join(f"{directory}/", f"{input_file}.jpg"))
    image_width, image_height = image.size

    
    
    # Resize the image to half its original size
    resized_image = image.resize((int(ratio * image_width), int(ratio * image_height)), Image.LANCZOS)
    
    # Save the resized image
    resized_image.save(os.path.join(f"{directory}/", f"{input_file.split(suffix)[0]}_{int(100*ratio)}.jpg"), "JPEG", quality=quality)
    
    return resized_image


########
def resize_jpeg_image_height(directory: str, input_file: str, new_height: int=2048, quality: int=100, suffix: str="") -> Image.Image:
    """
    Resize a JPEG image to half its original size.
    Args:
        directory (str): The directory where the JPEG file is located.
        input_file (str): The name of the JPEG file (without extension).
    Returns:
        Image: The resized JPEG image.
    """
    # Check if the input file exists
    if not os.path.exists(os.path.join(f"{directory}/", f"{input_file}.jpg")):
        print(f"File {input_file}.jpg does not exist.")

    # Load the image
    image = Image.open(os.path.join(f"{directory}/", f"{input_file}.jpg"))
    image_width, image_height = image.size

    new_width = int(image_width * (new_height / image_height))
    
    # Resize the image to half its original size
    resized_image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
    
    # Save the resized image
    resized_image.save(os.path.join(f"{directory}/", f"{input_file.split(suffix)[0]}_{new_height}.jpg"), "JPEG", quality=quality)
    
    return resized_image


########
def resize_jpeg_image(directory: str, input_file: str, new_width: int=1024, new_height: int=2048, quality: int=100, suffix: str="") -> Image.Image:
    """
    Resize a JPEG image to half its original size.
    Args:
        directory (str): The directory where the JPEG file is located.
        input_file (str): The name of the JPEG file (without extension).
    Returns:
        Image: The resized JPEG image.
    """
    # Check if the input file exists
    if not os.path.exists(os.path.join(f"{directory}/", f"{input_file}.jpg")):
        print(f"File {input_file}.jpg does not exist.")

    # Load the image
    image = Image.open(os.path.join(f"{directory}/", f"{input_file}.jpg"))
    image_width, image_height = image.size

    print(new_width, new_height)

    if new_width <= 0 and new_height <= 0:
        print("Both new_width and new_height are <= 0. No resizing will be done.")
        return image
    elif new_width == 0 and new_height > 0:
        new_width = int(image_width)
    elif new_height == 0 and new_width > 0:
        new_height = int(image_height)
    
    # Resize the image to half its original size
    resized_image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
    
    # Save the resized image
    resized_image.save(os.path.join(f"{directory}/", f"{input_file.split(suffix)[0]}_{new_height}.jpg"), "JPEG", quality=quality)
    
    return resized_image


########
def sharpen_text(directory: str, input_file: str, radius: float=1.0, strength: float=150, threshold: float=3) -> Image.Image:
    """
    Apply specialized text sharpening to improve legibility.
    Uses unsharp mask filter which is well-suited for text.
    
    Args:
        image: PIL Image to process
        strength: Sharpening strength (1.0-3.0 recommended)
        
    Returns:
        Sharpened PIL Image
    """

    # Check if the input file exists
    if not os.path.exists(os.path.join(f"{directory}/", f"{input_file}.jpg")):
        print(f"File {input_file}.jpg does not exist.")

    # Load the image
    image = Image.open(os.path.join(f"{directory}/", f"{input_file}.jpg"))
    
    # Convert to grayscale if color isn't needed
    if image.mode == "RGB":
        image = image.convert("L")
        
    # Apply unsharp mask filter with parameters optimized for text
    # radius: smaller radius works better for text
    # percent: amount of sharpening (higher = stronger effect)
    # threshold: minimum brightness change to apply sharpening
    sharpened_image = image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=strength, threshold=threshold)
    )

    sharpened_image.save(os.path.join(f"{directory}/", f"{input_file}_sharpened.jpg"), "JPEG", quality=100)
    
    return sharpened_image


########
# Function to convert a PIL Image to bytes
def convert_pil_image_to_bytes(pil_image: Image.Image, format: str="JPEG", quality: int=80) -> bytes:
    """
    Convert a PIL Image to bytes.
    Args:
        image (Image): The PIL Image to convert.
    Returns:
        bytes: The image in bytes format.
    """
    with io.BytesIO() as output:
        pil_image.save(output, format=format, quality=quality)
        return output.getvalue()


########
# Load image from a file
def load_image_from_file(file_path: str) -> Image.Image:
    """
    Load an image from a file.
    Args:
        file_path (str): The path to the image file.
    Returns:
        Image: The loaded image.
    """
    return Image.open(file_path)


########
def plot_solution_upper_bounds(pil_image: Image.Image, solution_upper_bounds: List[int], create_copy: bool = True) -> Image.Image:
    """
    Plots horizontal lines on an image at specified vertical positions.
    
    Args:
        pil_image: The PIL Image object to draw on
        solution_upper_bounds: A list of vertical positions (y-coordinates) for horizontal lines
        create_copy: If True, creates a copy of the image before drawing. If False, draws on the original.
    
    Returns:
        The image with drawn lines (either a copy or the original)
    """
    # Create a copy if requested (to avoid modifying the original)
    if create_copy:
        im = pil_image.copy()
    else:
        im = pil_image
        
    width, height = im.size
    
    # Create a drawing object
    draw = ImageDraw.Draw(im)
    
    # Red color for all lines
    color = "#FF0000"

    # Iterate over the solution upper bounds
    for k1, solution_upper_bound in enumerate(solution_upper_bounds):
        # Convert normalized coordinates to absolute coordinates
        x1 = 0
        x2 = width
        y1 = solution_upper_bound
        
        abs_y1 = int(y1 / 1000 * height)
        abs_x1 = int(x1 / 1000 * width)
        abs_x2 = int(x2 / 1000 * width)

        # Draw the horizontal line
        draw.line((abs_x1, abs_y1, abs_x2, abs_y1), fill=color, width=4)

    # Display the image
    display(im)
    
    # Return the image (either the modified original or the copy)
    return im

