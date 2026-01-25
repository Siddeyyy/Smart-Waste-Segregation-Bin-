import cv2
import time
import torch
import timm
import threading
import numpy as np
from PIL import Image
from torchvision import transforms
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library

# --- CONFIGURATION ---
MODEL_PATH = 'MobileNetV4_model_latest.pth'
MODEL_ARCH = 'mobilenetv4_conv_small'
LABELS = ['Biodegradable', 'Non-Biodegradable']

# --- GPIO SETUP (Raspberry Pi) ---
GPIO.setmode(GPIO.BCM)  # Use BCM Pin numbering
GPIO.setwarnings(False)

# Define Pins
LED_GREEN = 17   # Pin 11
LED_RED = 27     # Pin 13
LED_YELLOW = 22  # Pin 15

# Setup Pins as Output
GPIO.setup(LED_GREEN, GPIO.OUT)
GPIO.setup(LED_RED, GPIO.OUT)
GPIO.setup(LED_YELLOW, GPIO.OUT)

# Threshold for "Black Box" detection (0-255)
# If the average brightness is below this, we assume the bin is empty.
# Adjust this value: Increase if it thinks dark objects are empty; decrease if it thinks empty is an object.
EMPTY_THRESHOLD = 50 

def set_leds(color):
    """
    Helper to turn on one LED and turn off the others.
    color: 'green', 'red', 'yellow', or 'off'
    """
    # Turn all off first
    GPIO.output(LED_GREEN, GPIO.LOW)
    GPIO.output(LED_RED, GPIO.LOW)
    GPIO.output(LED_YELLOW, GPIO.LOW)

    if color == 'green':
        GPIO.output(LED_GREEN, GPIO.HIGH)
    elif color == 'red':
        GPIO.output(LED_RED, GPIO.HIGH)
    elif color == 'yellow':
        GPIO.output(LED_YELLOW, GPIO.HIGH)

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Lower resolution slightly to help RPi4 performance
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

def is_empty_black_box(frame):
    """
    Checks if the image is mostly dark (the black box bottom).
    Returns True if empty, False if an object is likely present.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate average pixel intensity
    avg_brightness = np.mean(gray)
    
    # Debug print (uncomment to calibrate your threshold)
    # print(f"Brightness: {avg_brightness}") 
    
    return avg_brightness < EMPTY_THRESHOLD

def main():
    print("Loading MobileNetV4... (Please wait)")
    device = torch.device('cpu') 
    
    try:
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=len(LABELS))
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error: {e}")
        return

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    stream = ThreadedCamera(0).start()
    time.sleep(1.0)
    print("Running Inference...")
    
    prev_time = 0

    try:
        while True:
            frame = stream.read()
            if frame is None: continue

            display_text = ""
            display_color = (255, 255, 255)

            # --- 1. CHECK FOR EMPTY BLACK BOX FIRST ---
            if is_empty_black_box(frame):
                display_text = "Empty (Black Box)"
                display_color = (0, 255, 255) # Yellow Text
                set_leds('yellow')
            
            # --- 2. RUN AI MODEL IF NOT EMPTY ---
            else:
                # Prepare Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)

                idx = predicted_idx.item()
                conf_score = confidence.item()
                
                try:
                    label_text = LABELS[idx]
                except IndexError:
                    label_text = "Unknown"

                # LED Logic based on Classification
                if label_text == 'Biodegradable':
                    display_color = (0, 255, 0) # Green Text
                    set_leds('green')
                else:
                    display_color = (0, 0, 255) # Red Text
                    set_leds('red')

                display_text = f"{label_text} ({int(conf_score*100)}%)"

            # --- DISPLAY STATS ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, display_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 3)

            cv2.imshow("Smart Bin - RPi4", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        print("Cleaning up...")
        stream.stop()
        cv2.destroyAllWindows()
        set_leds('off') # Turn off all LEDs
        GPIO.cleanup()  # Release GPIO pins

if __name__ == "__main__":
    main()