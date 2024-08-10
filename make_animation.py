import glob
import re
from PIL import Image

# Path to the folder containing the PNG files
folder_path = './plots/'

# Function to extract the numerical part of the filename for correct sorting
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return list(map(int, numbers))

# Use glob to get all the PNG files in the folder, sorted by filename numerically
image_files = sorted(glob.glob(f'{folder_path}/*.png'), key=numerical_sort)

# Open the images and save them in a list
images = [Image.open(image) for image in image_files]

# Save the images as an animated GIF
images[0].save('animation.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,  # Duration between frames in milliseconds
               loop=0)        # 0 means loop forever


