from models.yolov8 import YOLOv8
from models.kalman_filter import KalmanTracker
from models.data_association import associate_detections_to_tracks
from models.deep_appearance_descriptor import DeepAppearanceDescriptor
import numpy as np
import cv2

def draw_detections(image, detections):
    for detection in detections:
        bbox = detection['bbox']
        cls = detection['class']
        conf = detection['confidence']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{cls}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

class ObjectTracker:
    def __init__(self, weights_path):
        self.detector = YOLOv8(weights_path)
        self.tracks = []
        self.next_id = 1
        self.descriptor = DeepAppearanceDescriptor()

    def update(self, image):
        detections = self.detector.detect(image)
        for track in self.tracks:
            track.predict()

        associations = associate_detections_to_tracks(image, detections, self.tracks, self.cost_metric)

        for d, t in associations:
            self.tracks[t].update(np.array(detections[d]['bbox'][:2]))

        for d in detections:
            if all(d not in associations for _, d in associations):
                new_track = KalmanTracker()
                new_track.update(np.array(d['bbox'][:2]))
                new_track.id = self.next_id
                self.next_id += 1
                self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.kf.x[2] > 0 and t.kf.x[3] > 0]
        image_with_detections = draw_detections(image, detections)
        return image_with_detections

    def cost_metric(self, image, detection, track):
        det_feat = self.descriptor.extract_features(image, detection['bbox'])
        track_feat = self.descriptor.extract_features(image, track.state)
        da = self.descriptor.cosine_distance(det_feat, track_feat)
        dk = np.linalg.norm(np.array(detection['bbox'][:2]) - track.state)
        lambda_ = 0.5
        return lambda_ * dk + (1 - lambda_) * da