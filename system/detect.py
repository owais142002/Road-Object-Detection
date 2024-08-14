import cv2
from tracker import ObjectTracker

def main():
    cap = cv2.VideoCapture('./test.mp4')
    tracker = ObjectTracker('./detector_model_file/default.pt')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_detections = tracker.update(frame)

        for track in tracker.tracks:
            x, y = map(int, track.state)
            cv2.rectangle(frame_with_detections, (x, y), (x + 50, y + 50), (0, 255, 0), 2)
            cv2.putText(frame_with_detections, str(track.id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Frame', frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()