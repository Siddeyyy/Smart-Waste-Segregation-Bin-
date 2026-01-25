import cv2
import time
import threading
from ultralytics import YOLO

# --- CONFIGURATION ---
# POINT THIS TO YOUR NEW TFLITE FILE
# It might be named 'best_float16.tflite' or 'best_full_integer_quant.tflite'
MODEL_PATH = 'best_full_integer_quant.tflite' 
CONF_THRESHOLD = 0.45       # Slightly lower threshold for quantized models
INFERENCE_SIZE = 320        # Keep small for max FPS

# --- OPTIMIZED CAMERA CLASS ---
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
    print(f"Loading TFLite Model: {MODEL_PATH}...")
    try:
        # Ultralytics handles TFLite loading automatically!
        model = YOLO(MODEL_PATH, task='detect') 
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you copy the .tflite file correctly?")
        return

    # Start Camera
    stream = ThreadedCamera(0).start()
    time.sleep(1.0) # Warmup

    prev_time = 0
    print("Starting Loop. Press 'q' to exit.")

    while True:
        frame = stream.read()
        if frame is None: continue

        # --- INFERENCE ---
        # The 'imgsz' is critical for speed. 320 is standard for Pi optimization.
        results = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_SIZE, verbose=False, stream=True)

        status_text = "Scanning..."
        color = (255, 255, 255)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class Name
                cls_id = int(box.cls[0])
                # Note: TFLite models sometimes lose internal names, 
                # but Ultralytics usually preserves them.
                if model.names:
                    cls_name = model.names[cls_id]
                else:
                    # Fallback if names are lost in conversion
                    cls_name = "Class " + str(cls_id)

                # Logic for Biodegradable
                if cls_name == 'Biodegradable':
                    color = (0, 255, 0) # Green
                elif cls_name == 'Non-Biodegradable':
                    color = (0, 0, 255) # Red
                
                status_text = cls_name

                # Draw Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                cv2.putText(frame, f"{cls_name} {box.conf[0]:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- FPS COUNTER ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Display
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("TFLite Optimized - YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()