import cv2
import numpy as np
import os

# Directory for dataset
DATASET_DIR = "path/to/your/dataset"  

# Step 1: Capture Gesture Image or Load Pre-Captured Image
def capture_image():
    cap = cv2.VideoCapture(0)
    print("Press 's' to save the image and exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        cv2.imshow('Capture Gesture', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
            cv2.imwrite('captured_gesture.jpg', frame)
            print("Image saved as 'captured_gesture.jpg'")
            break
    cap.release()
    cv2.destroyAllWindows()

# Step 2: Pre-Processing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

# Step 3: Feature Extraction using Canny Edge Detection
def edge_detection(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)  # Adjust thresholds as needed
    return edges

# Step 4: Template Matching
def template_matching(edge_img, dataset_dir):
    best_match = None
    best_score = float('inf')
    for template_file in os.listdir(dataset_dir):
        template_path = os.path.join(dataset_dir, template_file)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        resized_template = cv2.resize(template, edge_img.shape[::-1])  # Resize template to match edge image
        diff = cv2.absdiff(edge_img, resized_template)
        score = np.sum(diff)  # Sum of absolute differences (SAD)
        if score < best_score:
            best_score = score
            best_match = template_file

    return best_match, best_score

# Step 5: Display Result
def display_result(match_file):
    if match_file:
        print(f"Recognized Gesture: {match_file}")
    else:
        print("No matching gesture found.")

# Complete Workflow
def main():
    # Step 1: Capture or Load an Image
    capture_image()
    image_path = "captured_gesture.jpg"
    
    # Step 2: Pre-Processing
    gray_img = preprocess_image(image_path)
    
    # Step 3: Edge Detection
    edge_img = edge_detection(gray_img)
    
    # Step 4: Template Matching
    match_file, score = template_matching(edge_img, DATASET_DIR)
    
    # Step 5: Display Result
    display_result(match_file)
    
    # Optional: Show images
    cv2.imshow("Grayscale Image", gray_img)
    cv2.imshow("Edge Detection", edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
