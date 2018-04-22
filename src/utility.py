import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from cv_function import *
from matplotlib.lines import Line2D
import time

DEBUG = True

# Codes are adapted from hw3 and OpenCV tutorial on Feature Detection and Description:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

# INPUT:
# 		img1, img2: input RGB or grey images as numpy array
# 		K1, K2: camera intrinsic matrices
# OUTPUT:
#		matches: Nx4 matching coordinates
#		R,t: second camera extrinsics
def find_matching(img1, img2, K1, K2):
	# Initiate SIFT detector
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
		if m.distance < 0.8*n.distance:
			good.append(m)

	if len(good)<=10:
		print("Not enough feature matches are found - %d/%d".format(len(good),10))
		return []

	img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
	img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

	H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,ransacReprojThreshold=15.0)

	img1_pts_inliers = [[x,y] for i,[x,y] in enumerate(img1_pts) if mask[i]==1]
	img2_pts_inliers = [[x,y] for i,[x,y] in enumerate(img2_pts) if mask[i]==1]

	matches = np.concatenate([img1_pts_inliers, img2_pts_inliers], axis=1)
	R, t = find_camera_pose(matches, K1, K2)

	return matches, R, t

# matches: a N x 4 array where:
# 			matches(i,1:2) is a point (w,h) in the first image
# 			matches(i,3:4) is the corresponding point in the second image
def find_camera_pose(matches, K1, K2):
	(F, res_err) = fundamental_matrix(matches)
	E = K2.T @ F @ K1

	(R, t) = find_rotation_translation(E)

	P1 = K1 @ np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)

	# the number of points in front of the image planes for all combinations
	num_points = np.zeros([len(t), len(R)])
	errs = np.full([len(t), len(R)], np.inf)

	for ti in range(len(t)):
		t2 = t[ti]
		for ri in range(len(R)):
			R2 = R[ri]
			P2 = K2 @ np.concatenate([R2, t2[:, np.newaxis]], axis=1)
			(points_3d, errs[ti,ri]) = find_3d_points(matches, P1, P2)
			Z1 = points_3d[:,2]
			Z2 = (points_3d @ R2[2,:].T + t2[2])
			num_points[ti,ri] = np.sum(np.logical_and(Z1>0,Z2>0))
	(ti,ri) = np.where(num_points==np.max(num_points))
	print(f"Reconstruction error = {errs[ti[0],ri[0]]}")

	t2 = t[ti[0]]
	R2 = R[ri[0]]
	return R2, t2


def unit_test():
	name = 'house'
	data_dir = "./data/{}".format(name)

	img1 = cv2.imread(f"{data_dir}/{name}1.jpg", 0)
	img2 = cv2.imread(f"{data_dir}/{name}2.jpg", 0)

	K1 = scipy.io.loadmat(f"{data_dir}/{name}1_K.mat")["K"]
	K2 = scipy.io.loadmat(f"{data_dir}/{name}2_K.mat")["K"]

	start = time.time()
	matches, R, t= find_matching(img1, img2, K1, K2)
	duration = time.time() - start
	print(f"feature matching time={duration}")

	pt1 = np.concatenate((matches[:, 0:2], np.ones((matches.shape[0], 1))), axis=1)
	pt2 = np.concatenate((matches[:, 2:], np.ones((matches.shape[0], 1))), axis=1)

	# Verify t, R
	print(f"t={t}")
	print(f"R={R}")

	# Verify matches
	VISIBLE_RATIO= 1
	matches = matches[:int(matches.shape[0]*VISIBLE_RATIO)]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.imshow(np.concatenate([img1, img2], axis=1))
	plt.plot(matches[:, 0], matches[:, 1], "+r")
	plt.plot(matches[:, 2] + img1.shape[1], matches[:, 3], "+r")
	for i in range(matches.shape[0]):
		line = Line2D([matches[i, 0], matches[i, 2] + img1.shape[1]], [matches[i, 1], matches[i, 3]], linewidth=1,
					  color="r")
		ax.add_line(line)
	plt.show()

	# Verify 3D pose
	P1 = K1 @ np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)
	P2 = K2 @ np.concatenate([R, t[:, np.newaxis]], axis=1)
	points, _ = find_3d_points(matches, P1, P2)
	plot_3d(points, t)
	
