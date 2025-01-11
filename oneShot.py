## YOLO images input, txt output
# Yolo starting code
import os

from ultralytics import YOLO

# Load the modified YOLO model (make sure the path points to your customized model)
model = YOLO("yolod_best.pt")

# Specify the image or folder containing images for prediction
source = "our_images"  # Replace with your actual path

# Run inference with specified options
results = model.predict(
    source=source,
    save=True,        # Save annotated images or videos
    save_txt=True,    # Save results in a text file format with [class] [x_center] [y_center] [width] [height] [confidence]
    imgsz=640,        # Image size for inference (can be adjusted)
    conf=0.5, 
    # show = True,    # Confidence threshold (can be adjusted based on your needs)
    save_conf=True,   # Include confidence scores in the saved text files
    device='cpu'   # Use GPU for faster inference (change to 'cpu' if GPU is not available)
)

# Identify the directory where labels are saved
labels_dir = results[0].save_dir  # save_dir attribute provides the directory path

# Construct the path variable for the label files
labels_path = os.path.join(labels_dir, "labels")  # 'labels' is the folder containing the text files

print(f"Labels are saved in: {labels_path}")

## DepthMap images input, depth map output
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    # Hardcoded values
    img_path = 'our_images'  # Path to the images folder
    outdir = 'depth_output'  # Output directory
    encoder = 'vits'  # Encoder type ('vits', 'vitb', 'vitl', 'vitg')
    input_size = 518  # Input size for images
    pred_only = False  # Flag for prediction only
    grayscale = False  # Flag for grayscale depth map

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # DEVICE='cpu'

    # Encoder configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the depth model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the filenames from img_path (if it's a directory or file)
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Get the colormap for the depth visualization
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Process each image
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        # Read the image
        raw_image = cv2.imread(filename)

        # Infer depth map from the image
        depth = depth_anything.infer_image(raw_image, input_size)

        # Normalize depth map
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Apply grayscale or color map
        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Save the depth map image
        depth_map_filename = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '_depth.png')
        cv2.imwrite(depth_map_filename, depth)

    print("Depth map generation completed.")

## Crop Function, txt input , depth map input, cropped depth map output
import cv2
import numpy as np
import os

def extract_labels_from_txt(label_file_path):
   
    label_data = []
    with open(label_file_path, 'r') as f:
        for line in f:
            label = line.strip().split()
            label_data.append(label)
    return label_data

def extract_and_resize(image_path, label_data, output_size=(224, 224)):
   
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # List to store resized crops
    cropped_resized_images = []

    for label in label_data:
        class_id, x_center, y_center, box_width, box_height, confidence = map(float, label)

        # Convert from relative coordinates to absolute pixel values
        x_center_abs = int(x_center * w)
        y_center_abs = int(y_center * h)
        box_width_abs = int(box_width * w)
        box_height_abs = int(box_height * h)

        # Calculate the top-left and bottom-right corners of the bounding box
        x1 = max(0, x_center_abs - box_width_abs // 2)
        y1 = max(0, y_center_abs - box_height_abs // 2)
        x2 = min(w, x_center_abs + box_width_abs // 2)
        y2 = min(h, y_center_abs + box_height_abs // 2)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Resize the cropped image to the output size
        resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_AREA)
        cropped_resized_images.append(resized_image)

    return cropped_resized_images

def process_images_and_labels(image_folder, label_folder, output_folder, output_size=(224, 224)):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all image files in the image folder
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        
        # Remove '_depth' from the image file name to match the label file name
        base_name = os.path.splitext(image_file)[0].replace('_depth', '')
        label_file_path = os.path.join(label_folder, f"{base_name}.txt")
        
        # Check if the label file exists
        if os.path.exists(label_file_path):
            # Extract label data from the .txt file
            label_data = extract_labels_from_txt(label_file_path)
            
            # Extract and resize the images
            resized_images = extract_and_resize(image_path, label_data, output_size)
            
            # Save the resized images with the modified file name
            for idx, img in enumerate(resized_images):
                output_file_name = f"{base_name}depth{idx}.jpg"
                output_file_path = os.path.join(output_folder, output_file_name)
                cv2.imwrite(output_file_path, img)
                print(f"Saved: {output_file_path}")
        else:
            print(f"Label file not found for image: {image_file}")

# Example usage:
image_folder = 'depth_output'
label_folder = labels_path
output_folder = 'input_classifier'
output_size = (224, 224)  # Example fixed size for the classifier

# Process all images and their corresponding labels
process_images_and_labels(image_folder, label_folder, output_folder, output_size)


# yolo classifier cropped depth map input, new txt output

import os
from ultralytics import YOLO
import torch

# Load a custom trained model
model = YOLO("yolocNew.pt")

# Predict on an image
# Path to the folder containing images
img_folder = "input_classifier"

# Path to the folder where you want to save the label files
labels_folder = labels_path

# Ensure labels folder exists
os.makedirs(labels_folder, exist_ok=True)

# List all image files in the folder (you can modify the extensions as needed)
image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))]



for image_file in image_files:
    # Construct the full path to the image
    img_path = os.path.join(img_folder, image_file)

    # Predict on the image
    results = model(img_path)

    # Access probabilities (the 'probs' attribute is a Probs object)
    probs = results[0].probs.data  # This is a torch tensor containing class probabilities

    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(probs)

    # Map the index to class names
    names_dict = results[0].names
    predicted_class = names_dict[predicted_class_index.item()] 
    base_name = os.path.basename(img_path).split('depth')[0]

    txt_file_path = os.path.join(labels_folder, base_name + ".txt")

    # Append the predicted class to the text file
    with open(txt_file_path, 'a') as file:
        file.write(predicted_class + "\n")  # Append predicted class followed by a newline

    print(f"Predicted class '{predicted_class}' appended to {txt_file_path}")


import cv2
import numpy as np
import os

# Define class names
names = ['AxeHead', 'Grenade', 'Blade', 'Gun', 'fire', 'other', 'smoke']

# Paths for images, labels, and output
image_folder = "our_images"
labels_folder = labels_path
output_folder = "labeled_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for image_file in os.listdir(image_folder):
    if not image_file.endswith(('.jpg', '.png', '.jpeg')):
        continue

    # Paths for the image and label files
    image_path = os.path.join(image_folder, image_file)
    txt_file_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + ".txt")

    # Skip if the label file doesn't exist
    if not os.path.exists(txt_file_path):
        print(f"Warning: No label file found for {image_file}")
        continue

    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Read label file lines
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # Process bounding boxes and their labels
    bbox_data = []
    for line in lines:
        if line.strip().isdigit() or line.strip() in ["real", "fake"]:
            # Store label for the following bounding boxes
            label = line.strip()
            for bbox in bbox_data:
                # Convert bbox relative coordinates to absolute pixel values
                index = int(bbox[0])
                x_center, y_center, box_width, box_height = map(float, bbox[1:5])
                x_center_abs, y_center_abs = int(x_center * w), int(y_center * h)
                box_width_abs, box_height_abs = int(box_width * w), int(box_height * h)
                
                x1, y1 = max(0, x_center_abs - box_width_abs // 2), max(0, y_center_abs - box_height_abs // 2)
                x2, y2 = min(w, x_center_abs + box_width_abs // 2), min(h, y_center_abs + box_height_abs // 2)
                
                # Set color and label text
                class_name = names[index]
                color = (0, 255, 0) if label == "real" else (0, 0, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {class_name}"
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Clear bbox data for next label
            bbox_data.clear()
        else:
            # Store bbox line data until label line is found
            bbox_data.append(line.strip().split())

    # Save the labeled image
    output_image_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + "_labeled.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Labeled image saved to {output_image_path}")

