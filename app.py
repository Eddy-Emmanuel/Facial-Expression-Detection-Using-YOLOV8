import cv2
import supervision as sv
from ultralytics import YOLO

def run_app():
    model = YOLO(model="best.pt")
    detector = sv.BoxAnnotator()
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        success, frame = camera.read()
        if success:
            model_prediction = model(frame)[0]
            detections = sv.Detections.from_ultralytics(model_prediction)
            
            labels = [f"{model.model.names[label]}-{confidence:.2f}" for label, confidence in zip(detections.class_id, detections.confidence)]

            annotated_frame = detector.annotate(scene=frame, detections=detections, labels=labels)
        
            cv2.imshow("Detection", annotated_frame)
            
            if cv2.waitKey(1) == ord("q"):
                break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()
