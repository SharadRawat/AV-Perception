import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt


path = pathlib.Path(__file__).parent.absolute()

image = cv2.imread(str(path) + "/road.png")
image_size = image.shape
image = cv2.resize(image, (image.shape[0], image.shape[1]))
image_size = image.shape
logo = cv2.imread(str(path) + "/indriya.jpg")
logo = cv2.resize(logo, (logo.shape[0], logo.shape[1]))
logo_size = logo.shape

img_coords = np.float32([[850, 1800, 1], [2400, 1800, 1], [2400, 2400, 1], [850, 2400, 1]])
logo_coords = np.float32([[[0, 0, 1], [450, 0, 1], [450, 850, 1], [0, 850, 1]]])
data = {}
data['points'] = []

for i in range(len(img_coords)):
    data['points'].append([img_coords[i][0], img_coords[i][1]])

points = np.vstack(data['points']).astype(float)

dst_pts = points
h, mask = cv2.findHomography(logo_coords, dst_pts)
im_temp = cv2.warpPerspective(logo, h, (image_size[1],image_size[0]))

cv2.fillConvexPoly(image, dst_pts.astype(int), 0, 16);

combined_image = image + im_temp



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(combined_image)
ax1.set_title('Original Image', fontsize=15)
# ax2.imshow(added_image)
# ax2.set_title('Warped Image', fontsize=15)
plt.show()







