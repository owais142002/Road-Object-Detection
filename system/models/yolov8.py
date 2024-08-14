from ultralytics import YOLO

class YOLOv8:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, image):
        results = self.model(image)
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                detections.append({'bbox': (x1, y1, x2, y2), 'class': int(cls), 'confidence': float(conf)})
        return detections