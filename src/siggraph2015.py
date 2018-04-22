import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
  return ret

def reconstructionError(matches, H):
  error = 0
  for match in matches:
    oriCoor = np.array([[matches[2]],[matches[3]],[1]])
    prjCoor = np.dot(H, oriCoor)
    prjCoor = prjCoor / prjCoor[2]
    trueCoor= np.array([[matches[0]], matches[1]])
    error += np.linalg.norm(prjCoor[0:2], trueCoor)
  error /= len(matches)
  return error

def findMatches(img1, img2):
  sift = cv2.xfeatures2d.SIFT_create()
  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)
  # FLANN parameters
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params,search_params)
  # Find nearest 2 neighbors
  feature_matches = flann.knnMatch(des1,des2,k=2)
  # store all the good matches as per Lowe's ratio test.
  good = []
  for m,n in feature_matches:
    if m.distance < 0.5*n.distance:
      good.append(m)
  img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
  img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

  H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,ransacReprojThreshold=5.0)

  img1_pts_inliers = [[x,y] for i,[x,y] in enumerate(img1_pts) if mask[i]==1]
  img2_pts_inliers = [[x,y] for i,[x,y] in enumerate(img2_pts) if mask[i]==1]

  matches = np.concatenate([img1_pts_inliers, img2_pts_inliers], axis=1)
  error = recontructionError(matches, H)

  return matches, H, error

def writeVideo(optFrames):
  #-------------------------------------------------------#
  # Writes a video given the desired frames               #
  #-------------------------------------------------------#
  print("to be implemented")

def Cr(frameI, frameJ):
  print("to be implemented")

def Co(frameI, homography):
  centerI = np.zeroes((2,1))
  centerI[0] = frameI.shape[0]
  centerI[1] = frameI.shape[1]
  centerJ = homographyTrans(centerI, homography)
  return np.linalg.norm(centerI - centerJ)

def Cm(frameI, frameJ):
  #-------------------------------------------------------#
  # Identical to Cm in the paper.                         #
  # d: length of the diagonal in pixels                   #
  # tau, gamma: empirical constant as mentioned in work   #
  #-------------------------------------------------------# 
  d = np.sqrt(frameI.shape[0] ** 2 + frameI.shape[1] **2)
  tau = 0.1 * d
  gamma = 0.5 * d
  matchingCost, homography = Cr(frameI, frameJ)
  if matchingCost >= tau:
    return gamma
  else:
    return Co(frameI, homography)

def generateVideo(frames, speedup, outName):
  #-------------------------------------------------------#
  # One function to wrap up all stuff                     #
  #-------------------------------------------------------#
  matches, H, err1 = findMatches(frames[0], frames[1])
  matches, H, err2 = findMatches(frames[0], frames[15])
  print(f"err1 is {err1}, err2 is {err2}")
#  VISIBLE_RATIO= 1
#  matches = matches[:int(matches.shape[0]*VISIBLE_RATIO)]
#  fig = plt.figure()
#  ax = fig.add_subplot(111)
#  plt.imshow(np.concatenate([frames[0], frames[10]], axis=1))
#  plt.plot(matches[:, 0], matches[:, 1], "+r")
#  plt.plot(matches[:, 2] + frames[0].shape[1], matches[:, 3], "+r")
#  for i in range(matches.shape[0]):
#    line = Line2D([matches[i, 0], matches[i, 2] + frames[0].shape[1]], [matches[i, 1], matches[i, 3]], linewidth=1,
#                   color="r")
#    ax.add_line(line)
#  plt.show()
  #optFrames = dtwFindOptFrames(frames, speedup)
  #writeVideo(optFrames, outName)
