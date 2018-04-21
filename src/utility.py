import sys
import numpy as np
import cv2
DEBUG = True

def readVideo(inputName):
  print(f"reading video {inputName}")
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
  if DEBUG:
    cv2.imshow("first frame", ret[0])
    cv2.imshow("last frame", ret[-1])
    cv2.waitKey(10000)
  return ret
