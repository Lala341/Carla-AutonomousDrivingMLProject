from PIL import Image
import os
import numpy as np

def load_images_from_folder(folder_path):
    images = []
    labels = []
    ind =0
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and ind<=2:
            # Load the image
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)

            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Extract semantic tags (assuming it's in the red channel)
            semantic_tags = img_array[:, :, 0]

            # Append the image and its semantic tags to the lists
            images.append(img_array)
            labels.append(semantic_tags)
            
            print("Label for image:", np.unique(semantic_tags))
            ind+=1

    return images, labels
def colorize_labels(label_array, color_mapping):
    # Create an RGB image from the labels using the color mapping
    colored_image = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

    for label, color in color_mapping.items():
        colored_image[label_array == label] = color

    return colored_image
def save_colored_images(images, labels, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define color mapping for each label
    #1  2  5  6  7  8  9 12 13 14 18 19 20 22
    color_mapping = {
   # 0: [0, 0, 0],        # None
   # 1: [128, 0, 0],      # Buildings
   # 2: [0, 128, 0],      # Fences
   # 3: [0, 0, 128],      # Other
   # 4: [255, 0, 0],      # Pedestrians
   # 5: [255, 255, 0],    # Poles
   # 6: [255, 255, 255],  # RoadLines
   # 7: [0, 0, 255],      # Roads
    14: [255, 255, 128],  # Sidewalks
  #  9: [0, 128, 0],      # Vegetation
   # 10: [255, 128, 0],   # Vehicles
  #  20: [128, 128, 128], # Walls
  #  12: [255, 128, 255]  # TrafficSigns
}


    for img, label in zip(images, labels):
        # Colorize the labels
        colored_image = colorize_labels(label, color_mapping)

        # Save the colored image
        output_path = os.path.join(output_folder, f"colored_{len(os.listdir(output_folder))}.png")
        Image.fromarray(colored_image).save(output_path)

# Provide the path to your folder containing images
folder_path = "data3/segmentation"

# Load images and labels
loaded_images, loaded_labels = load_images_from_folder(folder_path)
output_folder = "output"

# Load images and labels
loaded_images, loaded_labels = load_images_from_folder(folder_path)

# Save colored images
save_colored_images(loaded_images, loaded_labels, output_folder)
# Now you can check the labels and perform any analysis you need
for img, label in zip(loaded_images, loaded_labels):
    print("Label for image:", label)
