import sys
import numpy as np
import cv2
import utility as utl

def parseArgv(argv):
  if len(argv) != 3:
    print("This code should be run on python 3.6. Usage:")
    print("\tpython3.6 generateHyperlapse inputName speedupFactor")
    sys.exit()
    return None, None
  else:
    return argv[1], int(argv[2])

if __name__ == '__main__':
  inputName, speedup = parseArgv(sys.argv)
  frames = utl.readVideo(inputName)
