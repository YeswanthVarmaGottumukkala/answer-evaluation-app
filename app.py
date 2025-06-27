from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os


UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ← Move this AFTER defining UPLOAD_FOLDER

from utils.image_processor import extract_text_from_image
from utils.xlnet_model import get_model_prediction
from utils.xlnet_model import get_similarity_score
from werkzeug.utils import secure_filename
import shutil

app = FastAPI()

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/evaluate")
async def evaluate(
    request: Request,
    question: UploadFile = File(...),
    student_answer: UploadFile = File(...),
    reference_answer: UploadFile = File(...)
):
    try:
        files = {"question": question, "student": student_answer, "reference": reference_answer}
        paths = {}

        for key, file in files.items():
            if not allowed_file(file.filename):
                return {"error": f"Invalid file type: {file.filename}"}
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            paths[key] = file_path

        question_text = extract_text_from_image(paths["question"])
        student_text = extract_text_from_image(paths["student"])
        reference_text = extract_text_from_image(paths["reference"])

        score = get_model_prediction(question_text, student_text, reference_text)

        # 🎯 Bonus adjustment
        if score >= 75:
            score += 20
        elif 70 <= score < 75:
            score += 18
        elif 60 <= score < 65:
            score += 16
        else:
            score -= 10

        score = max(0, min(score, 100))

        
        print("📘 Question:", question_text)
        print("🧑 Student Answer:", student_text)
        print("📗 Reference Answer:", reference_text)
        print("🎯 Raw Score:", score)


        return {
            "success": True,
            "score": score,
            "question_text": question_text,
            "student_answer_text": student_text,
            "reference_answer_text": reference_text
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
