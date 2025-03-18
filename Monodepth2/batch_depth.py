import warnings
warnings.simplefilter("ignore")
import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from layers import disp_to_depth
from utils import readlines
from datasets import KITTIRAWDataset
from networks import *
from torchvision import transforms

# Pre-trained model paths
encoder_path = '/home/zxs/ml/monodepth2_640x192/encoder.pth'
depth_decoder_path = '/home/zxs/ml/monodepth2_640x192/depth.pth'

# Initialize encoder and depth decoder
encoder = ResnetEncoder(18, False)
depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4))

# Load model weights
loaded_dict_enc = torch.load(encoder_path)
encoder.load_state_dict(loaded_dict_enc, strict=False)  # Use strict=False to skip unmatched keys
encoder.eval()

loaded_dict_dec = torch.load(depth_decoder_path)
depth_decoder.load_state_dict(loaded_dict_dec)
depth_decoder.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
depth_decoder.to(device)

# Image preprocessing function
def preprocess_image(image_path, resize_width=640, resize_height=192):
    input_image = Image.open(image_path).convert("RGB")
    original_width, original_height = input_image.size
    input_image = input_image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)  # Convert to Tensor
    return input_image, original_width, original_height

# Create depth image
def createDepthImage(rgb_image_path):
    input_image, original_width, original_height = preprocess_image(rgb_image_path)
    input_image = input_image.to(device)

    # Perform inference to compute depth map
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]  # Get predicted disparity map
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False
    )  # Resize disparity map to original image size
    
    # Convert disparity map to depth map
    disp_resized_np = disp_resized.squeeze().cpu().numpy()  # Remove batch dimension and convert to NumPy array
    
    # Normalize to [0, 255] range
    if disp_resized_np.max() - disp_resized_np.min() > 0:
        disp_resized_np = (disp_resized_np - disp_resized_np.min()) / (disp_resized_np.max() - disp_resized_np.min())
    else:
        disp_resized_np = np.zeros_like(disp_resized_np)  # If the range is zero, generate a black image

    disp_resized_np = (disp_resized_np * 255).astype(np.uint8)
    
    # Convert to image
    depth_image = Image.fromarray(disp_resized_np, mode='L')
    
    # Duplicate depth map to 3 channels
    depth_image = np.stack((depth_image,) * 3, axis=-1)

    return depth_image

# Batch processing function to generate depth images
def batch_generate_depth_images(input_folder, output_folder):
    # Check if output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all RGB image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # Generate depth image using createDepthImage function
        depth_image = createDepthImage(image_path)
        
        # Convert depth image to pseudo-color
        depth_image_colored = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)  # Use JET colormap
        
        # Keep original filename but change extension to .png
        depth_image_filename = os.path.splitext(image_file)[0] + '.png'
        depth_image_path = os.path.join(output_folder, depth_image_filename)
        
        # Save colored depth image using OpenCV
        cv2.imwrite(depth_image_path, depth_image)

        print(f"Depth image saved to: {depth_image_path}")

# Input and output folder paths
input_folder = '/home/zxs/ml/KITTI/data/dataset/sequences/08/image_2'  # Folder containing RGB images
# input_folder = '/home/zxs/ml/KITTI-360/data_2d_raw/2013_05_28_drive_0010_sync_image_00/2013_05_28_drive_0010_sync/image_00/data_rect'
output_folder = '/home/zxs/ml/KITTI/data/dataset/sequences/08/depth'  # Folder for saving depth images
# output_folder = '/home/zxs/ml/KITTI-360/data_2d_raw/2013_05_28_drive_0010_sync_image_00/2013_05_28_drive_0010_sync/image_00/depth'

# Generate depth images in batch
batch_generate_depth_images(input_folder, output_folder)

