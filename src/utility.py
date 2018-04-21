import sys
import numpy as np
import cv2

def readVideo(inputName):
  capture = cv2.VideoCapture(inputName)
  if (not capture.isOpened()):
    print("input video file invalid")
    sys.exit()
  ret = []
  while (capture.isOpened()):
    valid, frame = capture.read()
    if valid:
      ret.append(frame)
    else:
      break
  capture.release()
  return ret
