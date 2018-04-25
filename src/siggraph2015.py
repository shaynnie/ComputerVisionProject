import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
import cv2
DEBUG = True

def writeVideo(optFrames):
  #-------------------------------------------------------#
  # Writes a video given the desired frames               #
  #-------------------------------------------------------#
  print("to be implemented")

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
      ret.append(frame[270:810, 480:1440, :])
    else:
      break
  capture.release()
  return ret

def reconstructionError(matches, H):
  error = 0
  for match in matches:
    oriCoor = np.array([[match[0]], [match[1]],[1]])
    prjCoor = H.dot(oriCoor)
    trueCoor= np.array([[match[2]], [match[3]]])
    prjCoor2D = np.array([[prjCoor[0,0]/prjCoor[2,0]], [prjCoor[1,0]/prjCoor[2,0]]])
    error += np.linalg.norm(prjCoor2D - trueCoor)
  error /= len(matches)
  return error

def Cr(img1, img2):
  #---------------------------------------------------------#
  # Cr = reconstruction error                               #
  #---------------------------------------------------------#
  #sift = cv2.xfeatures2d.ORB_create()
  # find the keypoints and descriptors with SIFT
  orb = cv2.ORB_create()
  #kp1, des1 = sift.detectAndCompute(img1,None)
  #kp2, des2 = sift.detectAndCompute(img2,None)
  kp1, des1 = orb.detectAndCompute(img1, None)
  kp2, des2 = orb.detectAndCompute(img2, None)
  # FLANN parameters
  #FLANN_INDEX_KDTREE = 0
  #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  #search_params = dict(checks=50)
  #flann = cv2.FlannBasedMatcher(index_params,search_params)
  # Find nearest 2 neighbors
  #feature_matches = flann.knnMatch(des1,des2,k=2)
  # store all the good matches as per Lowe's ratio test.
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
  matches = bf.match(des1, des2)
  #good = []
  #for m,n in feature_matches:
  #  if m.distance < 0.5*n.distance:
  #    good.append(m)
  img1_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
  img2_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)
  # img2_pts_i = H * img1_pts_i
  H, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC,ransacReprojThreshold=5.0)

  img1_pts_inliers = [[x,y] for i,[x,y] in enumerate(img1_pts) if mask[i]==1]
  img2_pts_inliers = [[x,y] for i,[x,y] in enumerate(img2_pts) if mask[i]==1]

  matches = np.concatenate([img1_pts_inliers, img2_pts_inliers], axis=1)
  error = reconstructionError(matches, H)

  return H, error

def Co(frameI, H):
  centerCoor = np.array([[frameI.shape[0]/2],[frameI.shape[1]/2],[1]])
  transCoor = H.dot(centerCoor)
  transCoor2D = np.array([[transCoor[0]/transCoor[2]], [transCoor[1]/transCoor[2]]])
  #print(f"centerCoor is \n {centerCoor}\n")
  #print(f"transCoor is \n {transCoor}\n")
  #print(f"transCoor2D is \n{transCoor2D}\n")
  #return np.linalg.norm(centerCoor[0:2] - transCoor2D)
  return math.sqrt((centerCoor[0] - transCoor2D[0])**2 + (centerCoor[1] - transCoor2D[1])**2)

def Cm(frameI, frameJ):
  #-------------------------------------------------------#
  # Identical to Cm in the paper.                         #
  # d: length of the diagonal in pixels                   #
  # tau, gamma: empirical constants as mentioned in work  #
  #-------------------------------------------------------# 
  d = np.sqrt(frameI.shape[0] ** 2 + frameI.shape[1] **2)
  tau = 0.1 * d
  gamma = 0.5 * d
  H, reconstructionErr = Cr(frameI, frameJ)
  if reconstructionErr >= tau:
    return gamma
  else:
    return Co(frameI, H)

def Cs(i, j, v):
  lambdaS = 3.0
  return min(lambdaS * abs(j - i - v), 100)

def Ca(h, i, j):
  return min((abs(j - i) - abs(i - h)) ** 2, 200)

def findMinValIdx(i, j, w, Dv):
  minVal = 100000000.0
  minIdx = 0
  lambdaA = 0.0
  for k in range(1, w):
    if (Dv[i - k, i] + lambdaA * Ca(i - k, i, j)) < minVal:
      minVal = (Dv[i - k, i] + lambdaA * Ca(i - k, i, j))
      minIdx = i - k
  return minVal, minIdx

def argMinMatrix(L, g, w, Dv):
  s = 0
  d = 0
  minCost= 100000000.0
  for i in range(L - g, L):
    for j in range(i + int(w/2), min(i + w + int(w/2), L)):
      #print(f"cost of block ({i}, {j}) is {Dv[i, j]}")
      if Dv[i,j] < minCost:
        minCost= Dv[i, j]
        s = i
        d = j
  return s, d

def generateVideo(frames, speedup, outName):
  #-------------------------------------------------------#
  # One function to wrap up all stuff                     #
  #-------------------------------------------------------#
  v = speedup
  g = v + 1
  w = v
  L = len(frames)
  print(f"{frames[0].shape}")
  Dv= np.ones([L, L]) * 10000000.0
  Tv= np.zeros([L, L])
  # Initialize Dv
  print("Initializing Dv")
  for i in range(0, g):
    print(f"\tdealing with frame {i}")
    for j in range(i + int(w/2), min(i + w + int(w/2), L)):
      CmVal = Cm(frames[i], frames[j])
      CsVal = Cs(i, j, v)
      #print(f"\tmatchiing frame {i} with frame {j}, Cm is {CmVal}, Cs is {CsVal}")
      Dv[i, j] = CmVal + CsVal
      #Dv[i, j] = Cm(frames[i], frames[j]) + lambdaS * Cs(i, j, v)
  
  # Populate Dv
  print("Populating Dv")
  for i in range(g, L):
    print(f"\tdealing with frame {i}")
    for j in range(i + int(w/2), min(i + w + int(w/2), L)):
      CmVal = Cm(frames[i], frames[j])
      CsVal = Cs(i, j, v)
      c = CmVal + CsVal
      #print(f"\tcomputing cost for frame {i} with frame {j}, Cm is {CmVal}, Cs is {CsVal}")
      minCost, argMin = findMinValIdx(i, j, w, Dv)
#      print(f"\tbest backtracking is from frame {argMin} to frame {i}, having cost {Dv[argMin, i]}")
      Dv[i, j] = c + minCost
      Tv[i, j] = argMin
  
  # Backtracing
  print("Backtracing")
  s, d = argMinMatrix(L, g, w, Dv)
  print(f"s is {s} d is {d}")
  p = [d]
  while s > g:
    p.insert(0, s)
    b = int(Tv[s,d])
    d = s
    s = b
  p.insert(0, s)
  print(f"p is {p}")

  out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (960, 1080))
  bfIdx = 0
  for idx in p:
    thisFrame = np.concatenate((frames[idx], frames[bfIdx]), axis=0)
    out.write(thisFrame)
    bfIdx += v
    if bfIdx >= L:
      bfIdx = L -1
  out.release()
  '''
  out = cv2.VideoWriter('outputNoAlg.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (960, 540))
  idx = 0
  while idx < L:
    out.write(frames[idx])
    idx += v
  out.release()
  '''
