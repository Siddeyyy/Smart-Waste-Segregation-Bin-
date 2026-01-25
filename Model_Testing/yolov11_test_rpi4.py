import cv2
import time
import threading
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'best.pt'      # Ensure this is in the same folder
CONF_THRESHOLD = 0.5        # Confidence (0.5 = 50%)
INFERENCE_SIZE = 320        # Kept at 320 for highest FPS on Pi 4

# --- OPTIMIZED CAMERA CLASS ---
# This runs the camera in a separate thread to prevent lag
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        (self.status, self.frame) = self.capture.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()

def main():
    print(f"Loading YOLO model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Start Camera
    stream = ThreadedCamera(0).start()
    time.sleep(1.0) # Allow camera to warm up

    prev_time = 0
    print("Starting Loop. Press 'q' to exit.")

    while True:
        frame = stream.read()
        if frame is None: continue

        # --- INFERENCE ---
        # stream=True and verbose=False make it faster
        results = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_SIZE, verbose=False, stream=True)

        status_text = "Scanning..."
        color = (255, 255, 255) # Default White

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the class name directly from your model
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] # 'Biodegradable' or 'Non-Biodegradable'
                
                # Set Color based on your specific classes
                if cls_name == 'Biodegradable':
                    color = (0, 255, 0) # Green
                elif cls_name == 'Non-Biodegradable':
                    color = (0, 0, 255) # Red
                
                status_text = cls_name

                # Draw Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                cv2.putText(frame, cls_name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- FPS COUNTER ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Display FPS and Status
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Smart Bin - YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()