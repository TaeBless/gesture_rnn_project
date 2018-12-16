import numpy as np
import cv2

def handsegment(frame):
    imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    lower = np.array([0, 133, 77])
    upper = np.array([255, 173, 127])
    imgHAND = cv2.inRange(imgYCC, lower, upper)

    return imgHAND
