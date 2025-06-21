import cv2
import numpy as np
import os
import easyocr
from PIL import Image

# Initialize EasyOCR reader once (outside function for efficiency)
# This will be loaded when the module is imported
try:
    reader = easyocr.Reader(['en'])
    ocr_available = True
except Exception as e:
    print(f"Warning: EasyOCR initialization failed: {str(e)}")
    print("Text extraction may not work properly.")
    ocr_available = False

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy
    """
    # Convert to grayscale if image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply denoising (less aggressive)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Apply adaptive thresholding (better for uneven lighting)
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return processed

def extract_text_from_image(image_path):
    """
    Extract text from an image file using EasyOCR
    """
    try:
        if not ocr_available:
            raise ValueError("EasyOCR is not available")
            
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Save the processed image temporarily
        temp_path = os.path.join(os.path.dirname(image_path), f"temp_processed_{os.path.basename(image_path)}")
        cv2.imwrite(temp_path, processed_image)
        
        # Extract text using EasyOCR
        results = reader.readtext(temp_path)
        
        # Delete temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Combine all detected text
        text = ' '.join([result[1] for result in results])
        text = text.strip()
        
        # If text is empty, try with the original image
        if not text:
            results = reader.readtext(image_path)
            text = ' '.join([result[1] for result in results])
            text = text.strip()
        
        return text
        
    except Exception as e:
        print(f"OCR failed: {str(e)}")
        # Return a placeholder message instead of raising an exception
        return "Text extraction failed. Please enter text manually."