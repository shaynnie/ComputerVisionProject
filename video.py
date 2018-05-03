import numpy as np
import cv2
from feature_matching import *

def main():
    cap = cv2.VideoCapture('./data/shires1.MOV')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    frameCount = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    # print (width, height, fps)
    print ("Video Information:")
    print (f"Height: {height} Width: {width} FrameCount: {frameCount} FPS: {fps}")

    name = 'house'
    data_dir = "./data/{}".format(name)
    # K1 = scipy.io.loadmat(f"{data_dir}/{name}1_K.mat")["K"]
    # K2 = scipy.io.loadmat(f"{data_dir}/{name}2_K.mat")["K"]
    # print (K2.shape)
    K = np.array([[-2768, 0, 1224], [0, -2768, 1632], [0, 0, 1]])
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
    allFrames = allFrames[:61]
    print (allFrames.shape)

    camera_R = []
    camera_t = []
    2D_tracks = {}
    3D_tracks = np.array([])

    start = time.time()
    for i in range(allFrames.shape[0]):
        min_err = np.finfo(np.float64).max
        best_R, best_t, best_next_frame, matches

        for j in range(i, i+fps/2, 1):
            if j >= allFrames.shape[0]:
                break   
            
            matches = find_matching(allFrames[i], allFrames[j])
            R, t, err = find_camera_pose(matches, K, K)
            if err < min_err:
                min_err = err
                best_t = t
                best_R = R
                best_next_frame = j

        print (f"Best next frame from {i}th frame is {j}th frame")
        i = best_next_frame - 1
        camera_t.append(best_t)
        camera_R.append(best_R)
        matchesa
        # if 2D_tracks.shape[0] == 0:
        #     2D_tracks.resize((matches.shape[0], 1))
        #     for (p_i, (x1, y1, x2, y2)) in enumerate(matches):

        #         # 2D_tracks[(x1, y1)].append((x2, y2))
        # else:
        #     for (p_i, (x1, y1, x2, y2)) in enumerate(matches):
        #         if np.where(2D_tracks[:-1] == (x1,y2))
        #         # for k, v in 2D_tracks:
        #         #     if v[-1] == (x1, y1):
        #         #         2D_tracks[k].append((x2, y2))


        # P1 = K @ np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)
        # P2 = K @ np.concatenate([best_R, best_t[:, np.newaxis]], axis=1)
        # points, _ = find_3d_points(matches, P1, P2)


            # if err < 10.:
            #     visualize(t, R, K, K, matches, allFrames[i], allFrames[j])

    duration = time.time() - start
    print(f"feature matching time={duration}")



if __name__ == '__main__':
    main()