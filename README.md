# Vehicle Image Quality Classification

## Data Observations

Some images in the dataset were not strictly vehicle related.
Instead of removing them (which would reduce the already limited data), the system was designed to handle such cases using:

- Higher confidence threshold (τ = 0.8)  
- Test-Time Augmentation (TTA)  
- Fusion with OpenCV features  

This improves robustness in real world scenarios.

---

## Overview

This project classifies vehicle images into three quality categories:

- Blur  
- Low Light  
- Good  

Additionally, the system determines whether an image should be **accepted or rejected** for downstream processing.

---

## Approach

The solution combines deep learning with classical computer vision techniques.

### Deep Learning Model
- EfficientNet-B0 fine tuned for 3 classes  

### Test Time Augmentation (TTA)
Multiple augmented versions of the same image are evaluated and averaged to improve stability.

### OpenCV Features
- Blur detection (Laplacian variance)  
- Brightness estimation  
- Edge detection  

### Fusion Strategy
Final Score = 0.7 × Deep Learning + 0.3 × OpenCV

---

## Thresholding

A configurable confidence threshold (τ = 0.8):

- If label = good and confidence > 0.8 → Accept  
- Otherwise → Reject  

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt

Run inference

python inference.py --input processed_data/val --output results.csv --threshold 0.8


⸻

Notes
	•	All dependencies are listed in requirements.txt.
	•	Update the input image path based on your local/project structure when running in VS Code or Codespaces.
	•	A small sample set of 3 images is included for quick testing.
	•	results_val.csv contains model predictions on the validation dataset and is used for evaluation metrics.

⸻

Sample Output

image_filename, predicted_label, confidence_score, status
image1.jpg, good, 0.99, Accept
image2.jpg, blur, 0.98, Reject

⸻

Evaluation

Metrics
	•	Blur → Precision: 1.00 | Recall: 0.99 | F1: 0.996
	•	Good → Precision: 0.99 | Recall: 1.00 | F1: 0.996
	•	Low Light → Precision: 1.00 | Recall: 1.00 | F1: 1.00

⸻

Inference Latency

Average inference time: 0.20 seconds per image 

⸻

Reliability
	•	Blur accepted: 0
	•	Low-light accepted: 0
	•	Good rejected: 5

The system is conservative and avoids passing low quality images.

⸻

Qualitative Analysis

Failure Analysis
	•	Blur → predicted as Good (1 case)
The misclassified image was not a vehicle but a structured object (e.g., a book).
Strong edge patterns caused it to be interpreted as sharp.

⸻

Edge Cases
	•	Uneven Lighting
    Brightness estimation and TTA improve robustness
	•	Mixed Quality Images
    TTA reduces confidence → safer rejection
	•	Structured Patterns
    Fusion helps reduce false good predictions

⸻

Conclusion

This system combines deep learning and classical methods to achieve reliable image quality classification, with strong performance and safe decision making.

⸻

Author

Derin Jacob Varghese

