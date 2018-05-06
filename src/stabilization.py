import numpy as np
import cv2
import sys
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
  '''
  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1,des2)
  img1_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
  img2_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)
  '''

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

def getSubsetFrames(frames, p):
  newFrames = []
  for idx in p:
    newFrames.append(frames[idx])
  return newFrames

def getAverage(outImg, j, k):
  coorListX = [-1, 1, -1, 1]
  coorListY = [1, -1, -1, 1]
  rgb = np.array([0,0,0])
  count = 0
  for idx in range(len(coorListX)):
    coorX = j + coorListX[idx]
    coorY = k + coorListY[idx]
    if coorX >= 0 and coorY >= 0 and coorX < outImg.shape[0] - 1 and coorY < outImg.shape[1] - 1:
      rgb = rgb + outImg[coorX, coorY, :]
      count += 1
  rgb = np.floor(rgb / count)
  rgb.reshape((1,1,3))
  rgb.astype(np.uint8)
#  print(f"\t {rgb.shape} {rgb}")
  return rgb

def getMask(img):
  r = img[:,:,0]
  g = img[:,:,1]
  b = img[:,:,2]
  mr= np.array(r != 0)
  mg= np.array(g != 0)
  mb= np.array(b != 0)
  m = np.logical_or(mr, mg)
  m = np.logical_or(m, mb)
  m = np.invert(m)
  m = np.reshape(m, (m.shape[0], m.shape[1], 1))
  m_3 = np.concatenate((m, m), axis = 2)
  m_3 = np.concatenate((m_3, m), axis = 2)
  return m

def stablizedVideoRigid(allFrames, p):
  #-----------------------------------------------------#
  # allFrames: ALL frames from the video                   #
  # p: list containing indices of optimal frame selected#
  #    from previous path planning                      #
  #-----------------------------------------------------#
  frames = getSubsetFrames(allFrames, p)
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
#  plt.plot(x,xTraj,'r--', x,yTraj,'b--')
#  plt.plot(x,xTrajSmooth,'r:', x,yTrajSmooth,'b:')
  plt.show()

  corners = np.array([[0,0,1], [0,rows,1], [cols,0,1], [cols,rows,1]])
  xLeft = 0
  xRight = cols
  yUp = 0
  yDown = rows
  # transforms[i]: rigid transform parameters from frame i + 1 to frame i
  for i in range(L-1):
    print(f"transforming frame {i}")
    #    [frame i + 1 + k to frame i + 1] + [frame i + 1 to frame i] + [shift of frame i]
    dx = transforms[i][0] + (xTrajSmooth[i] - xTraj[i])
    dy = transforms[i][1] + (yTrajSmooth[i] - yTraj[i])
    da = transforms[i][2] + (aTrajSmooth[i] - aTraj[i])
    A = computeEuclieanMatrix(dx,dy,da)
    outImg = cv2.warpAffine(frames[i], A, (cols,rows))
    mask = getMask(outImg)
#    mask = np.invert(np.array(outImg != 0))
    #######################################################
    thisIdx = p[i]
    losses  = []
    models  = []
    indices = []
    searchRange = 5
    for j in range(max(0, thisIdx - searchRange), min(len(allFrames) - 1, thisIdx + searchRange)):
      if j == thisIdx:
        continue
#      dx_1, dy_1, da_1 = getRigidTransform(allFrames[j], frames[i])
      dx_1, dy_1, da_1 = getRigidTransform(allFrames[j], frames[min(i + 1, L -1)])
      dx = dx_1 + transforms[i][0] + (xTrajSmooth[i] - xTraj[i])
      dy = dy_1 + transforms[i][1] + (yTrajSmooth[i] - yTraj[i])
      da = da_1 + transforms[i][2] + (aTrajSmooth[i] - aTraj[i])
      A = computeEuclieanMatrix(dx,dy,da)
      losses.append((dx**2 + dy**2 + 0.1 * da**2))
      models.append(A)
      indices.append(j)
      '''
      patch = cv2.warpAffine(allFrames[j], A, (cols, rows))
      patch = np.multiply(patch, mask)
      outImg = outImg + patch
      mask = np.invert(np.array(outImg != 0))
      '''
#    print(f"\tloss is {losses}")
#    print(f"\tmodel is {models}")
#    print(f"\tindex is {indices}")
#    print(f"loss is {losses}")
    sortedModels = [x for _, x in sorted(zip(losses, models))]
    sortedIndices= [x for _, x in sorted(zip(losses, indices))]
#    print(f"sorted model is {sortedModels}")
#    print(f"sorted index is {sortedIndices}")
    for modelIdx in range(len(sortedModels)):
      patch = cv2.warpAffine(allFrames[sortedIndices[modelIdx]], sortedModels[modelIdx], (cols, rows))
      patch = np.multiply(patch, mask)
      outImg = outImg + patch
      mask = getMask(outImg)
#      mask = np.invert(np.array(outImg != 0))      
    '''
    for j in range(searchRange):
      sourceIdx = thisIdx + j
      ### Postive
      if sourceIdx >= 0 and sourceIdx < len(allFrames):
        dx_1, dy_1, da_1 = getRigidTransform(allFrames[sourceIdx], frames[i])
      #dx_1, dy_1, da_1 = getRigidTransform(allFrames[j], frames[i])
        dx = dx_1 + transforms[i][0] + (xTrajSmooth[i] - xTraj[i])
        dy = dy_1 + transforms[i][1] + (yTrajSmooth[i] - yTraj[i])
        da = da_1 + transforms[i][2] + (aTrajSmooth[i] - aTraj[i])
        A = computeEuclieanMatrix(dx,dy,da)
        patch = cv2.warpAffine(allFrames[j], A, (cols, rows))
        patch = np.multiply(patch, mask)
        outImg = outImg + patch
        mask = np.invert(np.array(outImg != 0))
      sourceIdx = thisIdx - j
      if sourceIdx >= 0 and sourceIdx < len(allFrames):
        dx_1, dy_1, da_1 = getRigidTransform(allFrames[sourceIdx], frames[i])
      #dx_1, dy_1, da_1 = getRigidTransform(allFrames[j], frames[i])
        dx = dx_1 + transforms[i][0] + (xTrajSmooth[i] - xTraj[i])
        dy = dy_1 + transforms[i][1] + (yTrajSmooth[i] - yTraj[i])
        da = da_1 + transforms[i][2] + (aTrajSmooth[i] - aTraj[i])
        A = computeEuclieanMatrix(dx,dy,da)
        patch = cv2.warpAffine(allFrames[j], A, (cols, rows))
        patch = np.multiply(patch, mask)
        outImg = outImg + patch
        mask = np.invert(np.array(outImg != 0))
      ### Negative

    #######################################################
    '''
    for j in range(mask.shape[0]):
      for k in range(mask.shape[1]):
        if mask[j,k,0] == 1:
#          print(f"position {j}, {k} has no value {outImg[j,k,:]}")
          outImg[j,k,:] = getAverage(outImg, j, k)
#          print(f"\t{outImg[j,k,:]}")
    #cv2.imwrite(f'cropped_frame{i}.jpg',outImg)
    outFrames.append(outImg)
    '''
    outCorners = (A @ corners.T).T
    xl = max(outCorners[0,0], outCorners[1,0])
    xr = min(outCorners[2,0], outCorners[3,0])
    yu = max(outCorners[0,1], outCorners[2,1])
    yd = min(outCorners[1,1], outCorners[3,1])
    if (xl > 0.1*cols) or (xr < 0.9*cols) or (yu > 0.1*rows) or (yd < 0.9*rows):
      continue
    xLeft = max(xLeft, xl)
    xRight = min(xRight, xr)
    yUp = max(yUp, yu)
    yDown = min(yDown, yd)
    '''
  outFrames.append(frames[L-1])
  
  # cropping
  cropped = outFrames
#  for frame in outFrames:
#    cropped.append(frame)#frame[int(yUp):int(yDown), int(xLeft):int(xRight)])

  out = cv2.VideoWriter('outputStabilized.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (cropped[0].shape[1], cropped[0].shape[0]))
  for f in cropped:
    out.write(f)
  out.release()
  #cv2.imwrite('cropped_frame.jpg',cropped[0])
