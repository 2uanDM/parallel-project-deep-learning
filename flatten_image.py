import os 
import cv2
from PIL import Image

def flatten_image(image_path, output_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Resize image to 64 x 64 x 3
    img = cv2.resize(img, (64, 64))
    
    # Flatten image
    img = img.flatten()
    
    # Save image in text format, each pixel separated by a comma
    with open(output_path, 'w') as f:
        f.write(','.join(map(str, img)))


if __name__ == '__main__':
    flatten_image('image/original/cat2.jpg', 'image/flatten/cat2.txt')