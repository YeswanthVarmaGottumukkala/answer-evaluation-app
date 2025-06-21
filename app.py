from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from utils.image_processor import extract_text_from_image
from utils.xlnet_model import get_similarity_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['TIMEOUT'] = 300
# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        print("Request received at /evaluate")
        print("Files in request:", list(request.files.keys()))
        
        if 'question' not in request.files or 'student_answer' not in request.files or 'reference_answer' not in request.files:
            print("Missing files in request")
            return jsonify({'error': 'All three images are required'}), 400
            
        question_file = request.files['question']
        student_answer_file = request.files['student_answer']
        reference_answer_file = request.files['reference_answer']
        
        print(f"File names: {question_file.filename}, {student_answer_file.filename}, {reference_answer_file.filename}")
        
        # Check if files are valid
        for file in [question_file, student_answer_file, reference_answer_file]:
            if file.filename == '':
                print(f"Empty filename detected")
                return jsonify({'error': 'One or more files not selected'}), 400
            if not allowed_file(file.filename):
                print(f"Invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        # Save files
        question_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(question_file.filename))
        student_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(student_answer_file.filename))
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(reference_answer_file.filename))
        
        print(f"Saving files to: {question_path}, {student_path}, {reference_path}")
        
        question_file.save(question_path)
        student_answer_file.save(student_path)
        reference_answer_file.save(reference_path)
        
        print("Files saved successfully")
        
        # Extract text from images
        try:
            print("Extracting text from question image")
            question_text = extract_text_from_image(question_path)
            print(f"Question text: {question_text}")
            
            print("Extracting text from student answer image")
            student_answer_text = extract_text_from_image(student_path)
            print(f"Student answer text: {student_answer_text}")
            
            print("Extracting text from reference answer image")
            reference_answer_text = extract_text_from_image(reference_path)
            print(f"Reference answer text: {reference_answer_text}")
            
            # Get similarity score
            print("Calculating similarity score")
            similarity_score = get_similarity_score(question_text, student_answer_text, reference_answer_text)
            if similarity_score>=75:
                similarity_score += 20
            elif similarity_score>=70 and similarity_score<75:
                similarity_score +=18
            elif similarity_score<65 and similarity_score>=60:
                similarity_score +=16
            else:
                similarity_score -= 10

            print(f"Similarity score: {similarity_score}")
            
            return jsonify({
                'success': True,
                'score': similarity_score,
                'question_text': question_text,
                'student_answer_text': student_answer_text,
                'reference_answer_text': reference_answer_text
            })
        except Exception as e:
            import traceback
            print(f"Error in text extraction or scoring: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        import traceback
        print(f"Unexpected error in evaluate route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)