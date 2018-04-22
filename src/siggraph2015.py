import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
DEBUG = True
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
	if len(good)<=10:
		print("Not enough feature matches are found - %d/%d".format(len(good),10))
		return []
	img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
	img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

	H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,ransacReprojThreshold=5.0)

	img1_pts_inliers = [[x,y] for i,[x,y] in enumerate(img1_pts) if mask[i]==1]
	img2_pts_inliers = [[x,y] for i,[x,y] in enumerate(img2_pts) if mask[i]==1]

	matches = np.concatenate([img1_pts_inliers, img2_pts_inliers], axis=1)

	return matches, H

def writeVideo(optFrames):
  #-------------------------------------------------------#
  # Writes a video given the desired frames               #
  #-------------------------------------------------------#
  print("to be implemented")

def homographyTrans(coor, homography):
  oriCoor = np.ones((3,1))
  oriCoor[0:1] = coor
  newCoor = homography @ oriCoor
  retCoor = np.zeroes(2,1)
  retCoor[0] = newCoor[0]/newCoor[2]
  retCoor[1] = newCoor[1]/newCoor[2]
  return retCoor


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
  matches, H = findMatches(frames[0], frames[10])
  VISIBLE_RATIO= 1
  matches = matches[:int(matches.shape[0]*VISIBLE_RATIO)]
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.imshow(np.concatenate([frames[0], frames[10]], axis=1))
  plt.plot(matches[:, 0], matches[:, 1], "+r")
  plt.plot(matches[:, 2] + frames[0].shape[1], matches[:, 3], "+r")
  for i in range(matches.shape[0]):
    line = Line2D([matches[i, 0], matches[i, 2] + frames[0].shape[1]], [matches[i, 1], matches[i, 3]], linewidth=1,
                   color="r")
    ax.add_line(line)
  plt.show()
  #optFrames = dtwFindOptFrames(frames, speedup)
  #writeVideo(optFrames, outName)
