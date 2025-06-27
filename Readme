# 📝 Handwritten Answer Evaluation App using OCR + XLNet

An end-to-end machine learning application that extracts handwritten answers from images and evaluates them using a custom XLNet model trained on semantic similarity.

---

## 🚀 Live Demo

👉 [Try it on Hugging Face Spaces](https://yeswanthvarma-answer-evaluation-app.hf.space)

---

## 📌 Project Overview

This app takes images of handwritten answers and performs:
1. **OCR** to extract text from question, student answer, and reference answer.
2. **Similarity scoring** using a custom-trained XLNet model.
3. **Bonus logic** to adjust the final score based on thresholds.
4. **User interface** to upload images and view the evaluated score.

---

## 🧠 Core Technologies

- **FastAPI**: Web framework
- **EasyOCR**: For extracting handwritten text
- **Hugging Face Transformers**: XLNet model
- **Custom Training**: Trained on Q-A-R triplets
- **Docker**: For containerized deployment
- **Hugging Face Spaces**: Live hosted app

---

## 📦 Folder Structure

```
answer-evaluation-app/
├── app.py                     # FastAPI application
├── requirements.txt           # Dependencies
├── Dockerfile                 # For Hugging Face deployment
├── utils/
│   ├── image_processor.py     # EasyOCR + preprocessing
│   └── xlnet_model.py         # Model load and prediction
├── templates/
│   └── index.html             # Frontend HTML
├── static/
│   ├── css/style.css          # UI styling
│   ├── js/main.js             # JS for client interaction
│   └── uploads/               # Uploaded image storage
```

---

## 🔍 Model Details

- **Base Model**: `xlnet-base-cased` (Hugging Face)
- **Custom Trained On**: Question, student answer, reference answer, and human-evaluated scores
- **Loss**: MSELoss
- **Output**: Score from 0 to 100
---

## ✍️ Sample Use Case

- Upload 3 images:
  - Question image
  - Student handwritten answer
  - Reference answer
- App will:
  - Extract text
  - Score similarity using model
  - Apply bonus logic
  - Display final score and extracted text

---

## 🧑‍💻 Author

**Yeswanth Varma Gottumukkala**  
- Email: yeswanthvarma.g@gmail.com  

---

## 📜 License

This project is for educational and research purposes.  
Model and app are freely available to explore on Hugging Face Spaces.
