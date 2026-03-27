import cv2
import numpy as np

def get_opencv_scores(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_norm = min(1.0, blur_score / 200)

    brightness = gray.mean()
    light_norm = min(1.0, brightness / 100)

    edge_score = np.sum(cv2.Canny(gray, 100, 200) > 0)
    edge_norm = min(1.0, edge_score / 50000)

    return {
        "blur": 1 - blur_norm,
        "low_light": 1 - light_norm,
        "good": (blur_norm + light_norm + edge_norm) / 3
    }