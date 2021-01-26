import cv2
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
import pathlib

path = pathlib.Path(__file__).parent.absolute()
source_points = []

if __name__ == "__main__":
    img = cv2.imread(str(path) + "/road.png")
    image_size = img.shape

    parameters_file = open('dist_pickle.p', 'rb')
    db = p.load(parameters_file)
    mtx = db["mtx"]
    dist = db["dist"]

    undist = cv2.undistort(img, mtx, dist, None, mtx)

    pts1 = np.float32([[1000, 2050], [3200, 2050], [1000, 2464], [3200, 2464]])
    pts2 = np.float32([[0, 0], [2000, 0], [250, 2400], [2000, 2400]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(undist, M, (2100, 2400))

    img = cv2.resize()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(dst)
    ax2.set_title('Warped Image', fontsize=15)
    plt.show()



