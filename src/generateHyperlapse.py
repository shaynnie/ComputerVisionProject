import sys
import numpy as np
import cv2
import utility as utl
import siggraph2015 as alg

def parseArgv(argv):
  if len(argv) != 3:
    print("This code should be run on python 3.6. Usage:")
    print("\tpython3.6 generateHyperlapse inputName speedupFactor")
    sys.exit()
    return None, None
  else:
    videoName = argv[1].split('/')
    outName = videoName[-1] + "_" + argv[2] + "xSpeedup.mp4"
    return argv[1], int(argv[2]), outName

if __name__ == '__main__':
  inputName, speedup, outName = parseArgv(sys.argv)
  frames = utl.readVideo(inputName)
  alg.generateVideo(frames, speedup, outName)
