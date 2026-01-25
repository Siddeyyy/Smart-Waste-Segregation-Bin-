import os
import cv2
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from tqdm import tqdm  # Progress bar

# ==========================================
# 1. CONFIGURATION
# ==========================================
# 📂 FOLDER PATHS
TEST_DIR = r"C:\Users\siddh\Desktop\Project_Smart_Bin\Model_Testing\TEST\Test"        # <--- PUT YOUR 500+ IMAGES FOLDER NAME HERE
YOLO_OUT = r"C:\Users\siddh\Desktop\Project_Smart_Bin\Model_Testing\yolo_output"        # Output folder for YOLO
MOBI_OUT = r"C:\Users\siddh\Desktop\Project_Smart_Bin\Model_Testing\mobilenet_output"   # Output folder for MobileNet

# 🧠 MODEL PATHS
YOLO_PATH  = r"C:\Users\siddh\Desktop\Project_Smart_Bin\Model_Testing\best.pt" # Your YOLO Weights
MOBILE_PATH = r"C:\Users\siddh\Desktop\Project_Smart_Bin\Model_Testing\MobileNetV4_model_latest.pth"   # Your MobileNet Weights

# 🏷️ CLASSES
CLASSES = ['Biodegradable', 'Non-Biodegradable']

# Device Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Running on device: {device}")

# ==========================================
# 2. SETUP MODELS & DIRECTORIES
# ==========================================
# Create output directories if they don't exist
os.makedirs(YOLO_OUT, exist_ok=True)
os.makedirs(MOBI_OUT, exist_ok=True)

print("🚀 Loading Models... (This may take a moment)")

# --- LOAD YOLO ---
try:
    yolo_model = YOLO(YOLO_PATH)
    print("✅ YOLOv11 Loaded")
except Exception as e:
    print(f"❌ Error loading YOLO: {e}")
    exit()

# --- LOAD MOBILENET ---
try:
    mob_model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=2)
    mob_model.load_state_dict(torch.load(MOBILE_PATH, map_location=device))
    mob_model.to(device)
    mob_model.eval()
    print("✅ MobileNetV4 Loaded")
except Exception as e:
    print(f"❌ Error loading MobileNet: {e}")
    exit()

# MobileNet Preprocessing
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# ==========================================
# 3. PROCESS IMAGES LOOP
# ==========================================
# Get list of images (jpg, png, jpeg)
image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
print(f"\n📂 Found {len(image_files)} images in '{TEST_DIR}'. Starting processing...")

# Loop with progress bar
for img_name in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(TEST_DIR, img_name)
    
    # -----------------------------
    # A. RUN YOLOv11
    # -----------------------------
    try:
        results = yolo_model.predict(img_path, conf=0.25, verbose=False)
        for result in results:
            # Draw boxes
            yolo_img = result.plot()
            
            # Save to specific folder
            save_path = os.path.join(YOLO_OUT, img_name)
            cv2.imwrite(save_path, yolo_img)
    except Exception as e:
        print(f"⚠️ YOLO Failed on {img_name}: {e}")

    # -----------------------------
    # B. RUN MOBILENET
    # -----------------------------
    try:
        # Load Image
        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # Convert for Model (BGR -> RGB -> PIL -> Tensor)
        rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = mob_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            score, idx = torch.max(probs, 1)
            label = CLASSES[idx.item()]
            confidence = score.item() * 100
        
        # Draw Label on Image
        color = (0, 255, 0) if label == 'Biodegradable' else (0, 0, 255) # Green or Red
        text = f"{label}: {confidence:.1f}%"
        
        # Add black background strip for text readability
        h, w, _ = original_img.shape
        cv2.rectangle(original_img, (0, 0), (w, 40), (0,0,0), -1) 
        cv2.putText(original_img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Save to specific folder
        save_path = os.path.join(MOBI_OUT, img_name)
        cv2.imwrite(save_path, original_img)
        
    except Exception as e:
         print(f"⚠️ MobileNet Failed on {img_name}: {e}")

print("\n🎉 DONE! Processing Complete.")
print(f"📁 YOLO Results:   {os.path.abspath(YOLO_OUT)}")
print(f"📁 MobileNet Results: {os.path.abspath(MOBI_OUT)}")