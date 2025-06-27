document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const form = document.getElementById('evaluation-form');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const newEvaluationBtn = document.getElementById('new-evaluation-btn');
    const uploadSection = document.querySelector('.upload-section');
    const resultsSection = document.querySelector('.results-section');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const loadingStep = document.getElementById('loading-step');
    const flowSteps = document.querySelectorAll('.flow-step');
    
    // File upload areas
    setupFileUpload('question');
    setupFileUpload('student');
    setupFileUpload('reference');
    
    // Check if all files are uploaded to enable the evaluate button
    function checkAllUploaded() {
        const questionPreview = document.getElementById('question-preview');
        const studentPreview = document.getElementById('student-preview');
        const referencePreview = document.getElementById('reference-preview');
        
        if (questionPreview.style.display === 'block' && 
            studentPreview.style.display === 'block' && 
            referencePreview.style.display === 'block') {
            evaluateBtn.disabled = false;
        } else {
            evaluateBtn.disabled = true;
        }
    }
    
    // Setup file upload functionality
    function setupFileUpload(type) {
        const uploadArea = document.getElementById(`${type}-upload`);
        const fileInput = document.getElementById(`${type}-input`);
        const preview = document.getElementById(`${type}-preview`);
        
        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(fileInput, preview, uploadArea);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', () => {
            handleFileSelect(fileInput, preview, uploadArea);
        });
    }
    
    // Handle file selection
    function handleFileSelect(fileInput, preview, uploadArea) {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            
            // Check file type
            const fileType = file.type;
            if (fileType !== 'image/jpeg' && fileType !== 'image/png' && fileType !== 'image/jpg') {
                alert('Please upload an image file (JPEG, JPG or PNG)');
                return;
            }
            
            // Display preview
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                preview.style.display = 'block';
                uploadArea.style.display = 'none';
                checkAllUploaded();
            };
            reader.readAsDataURL(file);
        }
    }
    
    // Form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        updateFlowStep(0); // Upload step
        
        const formData = new FormData(form);
        
        // Ensure files are included (debugging)
    console.log("Files being sent:");
    console.log("Question file:", formData.get('question'));
    console.log("Student answer file:", formData.get('student_answer'));
    console.log("Reference answer file:", formData.get('reference_answer'));
        // Simulate the processing steps with delays for better UX
        setTimeout(() => {
            loadingStep.textContent = 'Preprocessing images...';
            updateFlowStep(1); // Preprocessing step
            
            setTimeout(() => {
                loadingStep.textContent = 'Evaluating answer...';
                updateFlowStep(2); // Evaluating step
                
                // Send data to server
                fetch('/evaluate', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    
                    
                    // Update score
                    updateScore(data.score);
                    
                    // Hide loading and show results
                    loadingOverlay.style.display = 'none';
                    uploadSection.style.display = 'none';
                    resultsSection.style.display = 'block';
                    updateFlowStep(3); // Score step
                })
                .catch(error => {
                    alert('Error: ' + error.message);
                    loadingOverlay.style.display = 'none';
                });
            }, 1500);
        }, 1500);
    });
    
    // Update the flow step indicators
    function updateFlowStep(activeIndex) {
        flowSteps.forEach((step, index) => {
            if (index < activeIndex) {
                step.classList.remove('active');
                step.classList.add('completed');
            } else if (index === activeIndex) {
                step.classList.add('active');
                step.classList.remove('completed');
            } else {
                step.classList.remove('active', 'completed');
            }
        });
    }
    
    // Update score display
    function updateScore(score) {
        const scoreValue = document.getElementById('score-value');
        const scoreProgress = document.querySelector('.score-progress');
        const scoreMessage = document.getElementById('score-message');
        
        // Convert score to percentage if it's a decimal
        const scorePercent = score <= 1 ? Math.round(score * 100) : score;
        
        // Update score text
        scoreValue.textContent = `${scorePercent}%`;
        
        // Update circle progress
        const circumference = 2 * Math.PI * 90; // 2πr where r=90
        const offset = circumference - (scorePercent / 100) * circumference;
        scoreProgress.style.strokeDashoffset = offset;

        
        // Set color based on score
        if (scorePercent >= 80) {
            scoreProgress.style.stroke = '#28a745'; // Success
            scoreMessage.textContent = 'Excellent! The answer matches the reference very well.';
        } else if (scorePercent >= 60) {
            scoreProgress.style.stroke = '#17a2b8'; // Info
            scoreMessage.textContent = 'Good job! The answer is quite similar to the reference.';
        } else if (scorePercent >= 40) {
            scoreProgress.style.stroke = '#ffc107'; // Warning
            scoreMessage.textContent = 'Average match. There is room for improvement.';
        } else {
            scoreProgress.style.stroke = '#dc3545'; // Danger
            scoreMessage.textContent = 'The answer differs significantly from the reference.';
        }
    }
    
    // New evaluation button
    newEvaluationBtn.addEventListener('click', function() {
        // Reset form
        form.reset();
        
        // Reset previews
        document.querySelectorAll('.preview').forEach(preview => {
            preview.style.display = 'none';
            preview.innerHTML = '';
        });
        
        // Show upload areas
        document.querySelectorAll('.upload-area').forEach(area => {
            area.style.display = 'block';
        });
        
        // Reset flow steps
        updateFlowStep(0);
        
        // Show upload section, hide results
        uploadSection.style.display = 'block';
        resultsSection.style.display = 'none';
        // Disable evaluate button
        evaluateBtn.disabled = true;
    });
});
