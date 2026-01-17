from ultralytics import YOLO
from supervision import Detections
import supervision as sv
import cv2

# Download thermal-trained YOLO model
model_path = "weights/yolo-thermal-approche2.pt"

model = YOLO(model_path)

def inference(image_path):
    cv_image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    model_output = model(cv_image, conf=0.6, verbose=False)
    detections = Detections.from_ultralytics(model_output[0])
    return cv_image, detections

def draw_detections(image, detections):
    # Create annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    # Create labels
    labels = [f"Person {i+1}" for i in range(len(detections))]

    # Draw bounding boxes and labels
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return annotated

# Use it
image, detections = inference('2.png')
print(f"Detected {len(detections)} people")

# Draw boxes and display
annotated_image = draw_detections(image, detections)
cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the result
cv2.imwrite('1_detected.jpg', annotated_image)