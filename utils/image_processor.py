import os
import cv2
import numpy as np
from PIL import Image
import easyocr

# ✅ Safe directory for OCR model storage
EASYOCR_DIR = "/tmp/.easyocr"
os.makedirs(EASYOCR_DIR, exist_ok=True)

# ✅ Set environment variable before import (optional but safe)
os.environ["EASYOCR_HOME"] = EASYOCR_DIR

# ✅ Initialize EasyOCR
try:
    reader = easyocr.Reader(['en'], model_storage_directory=EASYOCR_DIR)
    ocr_available = True
    print("✅ EasyOCR initialized.")
except Exception as e:
    print(f"❌ EasyOCR initialization failed: {str(e)}")
    ocr_available = False

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed

def extract_text_from_image(image_path):
    """
    Extract text from image using EasyOCR
    """
    print(f"📂 Reading image from: {image_path}")
    try:
        if not ocr_available:
            raise ValueError("EasyOCR not available")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        processed = preprocess_image(image)
        temp_path = os.path.join(os.path.dirname(image_path), f"temp_{os.path.basename(image_path)}")
        cv2.imwrite(temp_path, processed)

        results = reader.readtext(temp_path)
        os.remove(temp_path)

        text = ' '.join([res[1] for res in results]).strip()

        # Fallback to original if empty
        if not text:
            results = reader.readtext(image_path)
            text = ' '.join([res[1] for res in results]).strip()

        print("📝 Extracted Text:", text)
        return text or "Text extraction failed. Please enter text manually."

    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return "Text extraction failed. Please enter text manually."
