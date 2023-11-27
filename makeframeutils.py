import torch
import numpy as np
from pathlib import Path
import os
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Get num closest to 8
def cl8(num):
    rem = num % 8
    if rem <= 4:
        return round(num - rem)
    else:
        return round(num + (8 - rem))

def closest_lcm(n, div1, div2):
    # Find the LCM
    lcm = np.lcm(int(div1), int(div2))

    # Find the multiple of LCM closest to n
    lower_multiple = (n // lcm) * lcm  # Largest multiple of lcm less than or equal to n
    upper_multiple = lower_multiple + lcm  # Smallest multiple of lcm greater than n

    # Find which multiple is closer to n
    if n - lower_multiple > upper_multiple - n:
        return upper_multiple
    else:
        return lower_multiple
    
def normalize_size(images):
    refimage = images[0]
    refimage = refimage.resize((cl8(refimage.width), cl8(refimage.height)), Image.Resampling.LANCZOS)
    return_images = []
    for i in range(len(images)):
        if images[i].size != refimage.size:
            images[i] = images[i].resize(refimage.size, Image.Resampling.LANCZOS)
        return_images.append(images[i])
    return return_images

def constrain_image(image, max_width, max_height):
    width, height = image.size
    aspect_ratio = width / float(height)

    if width > max_width or height > max_height:
        if width / float(max_width) > height / float(max_height):
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        image = image.resize((cl8(new_width), cl8(new_height)), Image.Resampling.LANCZOS)

    return image

def padlist(lst, targetsize):
    if targetsize <= len(lst):
            return lst[:targetsize]

    last_elem = lst[-1]
    num_repeats = targetsize - len(lst)
    
    return lst + [last_elem] * num_repeats

def MakeGrid(images, rows, cols):
    widths, heights = zip(*(i.size for i in images))

    grid_width = max(widths) * cols
    grid_height = max(heights) * rows
    cell_width = grid_width // cols
    cell_height = grid_height // rows
    final_image = Image.new('RGB', (grid_width, grid_height))
    x_offset = 0
    y_offset = 0
    for i in range(len(images)):
        final_image.paste(images[i], (x_offset, y_offset))
        x_offset += cell_width
        if x_offset == grid_width:
            x_offset = 0
            y_offset += cell_height

    # Save the final image
    return final_image

def BreakGrid(grid, rows, cols):
    width = grid.width // cols
    height = grid.height // rows
    outimages = []
    for row in range(rows):
            for col in range(cols):
                left = col * width
                top = row * height
                right = left + width
                bottom = top + height
                current_img = grid.crop((left, top, right, bottom))
                outimages.append(current_img)
    return outimages

def ImgLabeler(img, text, size=72, color=(255,255,255)):
    font = ImageFont.truetype("arial.ttf", size)
    draw = ImageDraw.Draw(img)

    # Get text size
    text_size = draw.textlength(text, font=font)
    # Calculate x, y coordinates of the text
    x = (img.width - text_size) / 2
    y = (img.height - text_size) / 2

    # Position for the text, centered
    text_position = (x, y)

    # Draw the text onto the image
    draw.text(text_position, text, font=font, fill=color)
    
    return img

def load_and_preprocess(image_path):
    with Image.open(image_path) as img:
        return torch.tensor(np.array(img.convert('L')), device=device, dtype=torch.float32)

def compute_histogram(tensor):
    # Min and max values for grayscale images
    min_val, max_val = 0, 255
    # Compute the histogram by counting the number of occurrences within each bin
    hist = torch.histc(tensor, bins=255, min=min_val, max=max_val)
    return hist / tensor.numel()  # Normalize by the number of elements

def get_iterated_path(directory, base_filename, extension='.png'):
    # Construct the full file path
    count = 0
    while True:
        # Append a count to the filename if it's not the first file
        if count == 0:
            unique_filename = f"{base_filename}{extension}"
        else:
            unique_filename = f"{base_filename}_{count}{extension}"
        full_file_path = os.path.join(directory, unique_filename)

        # Check if a file with this name already exists
        if not os.path.exists(full_file_path):
            break  # Exit the loop once the image is saved
        count += 1
    
    return full_file_path

def CheckMakeDir(dir):
    try:
        if not Path.exists(Path(dir)):
            Path.mkdir(Path(dir), parents=True, exist_ok=True)
        return True, "Success"
    except:
        st_out = f"Error: could not locate nor create directory {dir}."
        print(st_out)
        return False, st_out

def apply_conditional_ema_pytorch(images: list[Image.Image], alpha: float, threshold: float) -> list[Image.Image]:

    processed_images = []
    ema_tensor = None
    to_tensor = ToTensor()
    to_pil = ToPILImage()

    for image in images:
        frame_tensor = to_tensor(image).to(device)

        if ema_tensor is None:
            # For the first frame, EMA is just the frame itself
            ema_tensor = frame_tensor.clone()
        else:
            # Calculate EMA for each subsequent frame
            deviation = torch.abs(frame_tensor - ema_tensor)
            update_mask = deviation > threshold  # Mask to identify where to update

            # EMA update
            ema_tensor = alpha * frame_tensor + (1 - alpha) * ema_tensor
            ema_tensor[update_mask] = frame_tensor[update_mask]  # Apply conditional update

        processed_images.append(to_pil(ema_tensor))

    return processed_images

def cat_to_pils(tensor):
    to_pil = ToPILImage()
    tensor = tensor.permute(0, 3, 1, 2)
    pils = [to_pil(tensor[i]) for i in range(tensor.shape[0])]
    return pils

def pil_to_tens(pimg):
    to_tensor = ToTensor()
    tensor = to_tensor(pimg).unsqueeze(0).permute(0, 2, 3, 1)
    return tensor

def get_grid_aspect(num_images: int, image_width: int, image_height: int) -> (int, int):

    if num_images == 0:
        return 0, 0

    min_diff = float('inf')
    best_layout = (1, num_images)

    if image_width > image_height:
        for cols in range(1, num_images + 1):
            rows = -(-num_images // cols)
            grid_width = cols * image_width
            grid_height = rows * image_height
            diff = abs(grid_width - grid_height)

            if diff < min_diff:
                min_diff = diff
                best_layout = (rows, cols)

            if cols > num_images / cols:
                break
    else:
        for rows in range(1, num_images + 1):
            cols = -(-num_images // rows)
            grid_width = cols * image_width
            grid_height = rows * image_height
            diff = abs(grid_height - grid_width)

            if diff < min_diff:
                min_diff = diff
                best_layout = (rows, cols)

            if rows > num_images / rows:
                break

    return best_layout