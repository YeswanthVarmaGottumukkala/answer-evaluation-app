FROM python:3.10-slim

#  Set all required environment variables
ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch
ENV EASYOCR_HOME=/tmp/.easyocr
ENV HOME=/tmp  
#  System dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*



#  Pre-create the EasyOCR directory
RUN mkdir -p /tmp/.easyocr && chmod -R 777 /tmp/.easyocr

# Set working directory
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
