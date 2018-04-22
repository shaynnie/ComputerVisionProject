import numpy as np
import cv2
from feature_matching import *

def main():
    cap = cv2.VideoCapture('shires1.MOV')
    # print (dir(cap))
    # if cap.isOpened(): 
    # get vcap property 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    frameCount = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    # print (width, height, fps)
    print ("Video Information:")
    print (f"Height: {height} Width: {width} FrameCount: {frameCount} FPS: {fps}")

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480);
    name = 'house'
    data_dir = "./data/{}".format(name)
    # K1 = scipy.io.loadmat(f"{data_dir}/{name}1_K.mat")["K"]
    # K2 = scipy.io.loadmat(f"{data_dir}/{name}2_K.mat")["K"]
    # print (K2.shape)
    K = np.array([[2828, 0, 1274], [0, 2842, 1627], [0, 0, 1]])
    print (K.shape)


    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = cv2.flip(gray, -1)
            # M = cv2.getRotationMatrix2D(center, 90., 1.)
            # rotated90 = cv2.warpAffine(gray, M, (height, width))

            # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('frame', 960,540)
            resized = cv2.resize(gray, (960, 540), interpolation = cv2.INTER_AREA)
            # cv2.imshow('frame',resized)
            frames.append(resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    # cv2.destroyAllWindows()

    allFrames = np.array(frames)
    print (allFrames.shape)
    for i in range(allFrames.shape[0]):
        start = time.time()
        # if i == 2: 
            # break
        for j in range(i+1, i+fps, 1):
            print (f"find matches between {i}th frame and {j}th frame")
            matches = find_matching(allFrames[i], allFrames[j])
            R, t = find_camera_pose(matches, K, K)

            # visualize(t, R, matches, allFrames[i], allFrames[j])

        duration = time.time() - start
        print(f"feature matching time={duration}")



if __name__ == '__main__':
    main()