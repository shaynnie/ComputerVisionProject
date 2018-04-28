import numpy as np
import cv2
import sys
import siggraph2015 as alg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage.filters import gaussian_filter1d

# Finds A: img2 = A * img1
def getRigidTransform(img1, img2):
	# find the keypoints and descriptors with SIFT
	sift = cv2.xfeatures2d.SIFT_create()
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
		if m.distance < 0.7*n.distance:
			good.append(m)
	img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
	img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

	# orb = cv2.ORB_create()
	# kp1, des1 = orb.detectAndCompute(img1,None)
	# kp2, des2 = orb.detectAndCompute(img2,None)

	# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# matches = bf.match(des1,des2)

	# img1_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
	# img2_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)

	A = cv2.estimateRigidTransform(img1_pts, img2_pts, False)
	if A is None:
		return 0,0,0
	dx = A[0,2]
	dy = A[1,2]
	da = np.arctan2(A[1,0], A[0,0])
	return dx, dy, da

def computeEuclieanMatrix(dx,dy,da):
	A = np.zeros((2,3))
	A[0,0] = np.cos(da)
	A[0,1] = -np.sin(da)
	A[1,0] = np.sin(da)
	A[1,1] = np.cos(da)
	A[0,2] = dx
	A[1,2] = dy
	return A

def stablizedVideoRigid(frames):
	outFrames = []
	transforms = []
	xTraj = []
	yTraj = []
	aTraj = []
	x = 0.0
	y = 0.0
	a = 0.0
	L = len(frames)
	curFrame = frames[0]
	cols = frames[0].shape[1]
	rows = frames[0].shape[0]

	for i in range(1,L):
		print(f"obtaining rigid transform from frame {i} to {i-1}")
		prevFrame = curFrame
		curFrame = frames[i]
		dx, dy, da = getRigidTransform(prevFrame, curFrame)
		transforms.append([dx, dy, da])
		x = x + dx
		y = y + dy
		a = a + da
		xTraj.append(x)
		yTraj.append(y)
		aTraj.append(a)

	SIGMA = 5
	xTrajSmooth = gaussian_filter1d(xTraj, sigma=SIGMA)
	yTrajSmooth = gaussian_filter1d(yTraj, sigma=SIGMA)
	aTrajSmooth = gaussian_filter1d(aTraj, sigma=SIGMA)

	x = np.arange(len(xTraj))
	plt.plot(x,xTraj,'r--', x,yTraj,'b--')
	plt.plot(x,xTrajSmooth,'r:', x,yTrajSmooth,'b:')
	plt.show()

	for i in range(L-1):
		print(f"transforming frame {i}")
		dx = transforms[i][0] + xTrajSmooth[i] - xTraj[i]
		dy = transforms[i][1] + yTrajSmooth[i] - yTraj[i]
		da = transforms[i][2] + aTrajSmooth[i] - aTraj[i]
		A = computeEuclieanMatrix(dx,dy,da)
		outImg = cv2.warpAffine(frames[i], A, (cols,rows))
		outFrames.append(outImg)

	outFrames.append(frames[L-1])

	out = cv2.VideoWriter('outputStabilized.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (cols, rows))
	for f in outFrames:
		out.write(f)
	out.release()

if __name__ == '__main__':
	if len(sys.argv) != 2:
	    print("Usage:python stabilization.py inputVideo")
	    sys.exit()
	sys.argv
	frames = alg.readVideo(sys.argv[1])
	stablizedVideoRigid(frames)


