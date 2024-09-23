import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image_with_labels(image_path, label_path, class_names=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return
    height, width, _ = image.shape

    # Read the label file
    if not os.path.exists(label_path):
        print(f"Error: Label file {label_path} does not exist.")
        return

    with open(label_path, 'r') as file:
        lines = file.readlines()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for line in lines:
        # Parse the label file
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Skipping invalid line in {label_path}: {line}")
            continue

        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

        x_center = int(x_center * width)
        y_center = int(y_center * height)
        bbox_width = int(bbox_width * width)
        bbox_height = int(bbox_height * height)

        top_left_x = int(x_center - bbox_width / 2)  # krai - nachalo / 2 -> centur
        top_left_y = int(y_center - bbox_height / 2)

        rect = patches.Rectangle((top_left_x, top_left_y), bbox_width, bbox_height, linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

        # Draw the class label
        if class_names and int(class_id) < len(class_names):
            label = class_names[int(class_id)]
            plt.text(top_left_x, top_left_y - 10, label, color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

# Example usage
image_dir = r"C:\Users\Gamer\farm_prototype\leaf_doctor\leaves_2-2\train\images"
label_dir = r'C:\Users\Gamer\farm_prototype\leaf_doctor\leaves_2-2\train\labels'
class_names = ['leaf', 'leaf', 'leaf']


for image_name in os.listdir(image_dir):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

        visualize_image_with_labels(image_path, label_path, class_names)
