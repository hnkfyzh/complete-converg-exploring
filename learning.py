import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import numpy as np
import copy
import math
import time
from window import Window
import gym
import sys
from Astar import AStar
import random
import os
import cv2
import matplotlib.pyplot as plt




def show_img(img):
    """
    Show an image or update the image being shown
    """

    # Show the first image of the environment
    imshow_obj = plt.imshow(img, interpolation='bilinear')



    # Let matplotlib process UI events
    # This is needed for interactive mode to work properly
    plt.pause(100)
def scale_binary_map(binary_map, scale_factor):
    # Use zoom function for nearest-neighbor interpolation
    scaled_map = zoom(binary_map, scale_factor, order=0, mode='nearest')

    # Threshold the scaled map to convert it back to a binary map
    scaled_map = (scaled_map > 0.5).astype(int)

    return scaled_map

# Assuming binary_map is your original binary map
map_img = Image.open("map4.pgm")
map_img = np.array(map_img)
height = int(map_img.shape[0] * 0.1)
width = int(map_img.shape[1] * 0.1)
map_img = cv2.resize(map_img, (height, width), interpolation=cv2.INTER_NEAREST)
ref_point = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_image = image.copy()
            cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


# 加载图像并设置窗口
image = map_img
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):
        if len(ref_point) == 2:
            roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("ROI", roi)
            cv2.imwrite('cropped_image.jpg', roi)
            cv2.waitKey(0)
        break

    elif key == 27:
        break

cv2.destroyAllWindows()
# binary_map = binary_map.T
binary_map = np.rot90(binary_map,3)

# Scale factor for halving the size

# Call the scale_binary_map function
scaled_map = binary_map

print("Original Map:")
print(binary_map)
print(scaled_map.shape)
# scaled_map = np.where(scaled_map == 1, 254, scaled_map)
# scaled_map = scaled_map.astype(np.uint8)
# img = cv2.cvtColor(np.array(scaled_map), cv2.COLOR_BGR2GRAY)
kernel = np.ones((1, 2), np.uint8)
kernel_2 = np.ones((20, 20), np.uint8)

# opened_image = cv2.morphologyEx(scaled_map, cv2.MORPH_OPEN, kernel)
# opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_OPEN, kernel)
# opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_OPEN, kernel)
# scaled_map = cv2.morphologyEx(scaled_map, cv2.MORPH_CLOSE, kernel)

# scaled_map = cv2.dilate(scaled_map, kernel, 1)#去噪点
scaled_map = cv2.erode(scaled_map, kernel_2, 1)#障碍物膨胀
# scaled_map = cv2.morphologyEx(scaled_map, cv2.MORPH_GRADIENT, kernel)

scale_factor = 0.1
scaled_map = scale_binary_map(scaled_map, scale_factor)
scaled_map = scaled_map[20:,:]
scaled_map = np.where(scaled_map == 1, 254, scaled_map)
scaled_map = scaled_map.astype(np.uint8)
scaled_map = cv2.copyMakeBorder(scaled_map, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
# scaled_map = cv2.line(scaled_map,(0,18),(198,18),0,2)
a = []
b = []
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "[%d,%d]" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(scaled_map, (x, y), 3, (255, 0, 0), thickness=-1)
        cv2.putText(scaled_map, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)  # 如果不想在图片上显示坐标可以注释该行
        cv2.imshow("image", scaled_map)
        print("[{},{}]".format(a[-1], b[-1]))  # 终端中输出坐标


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", scaled_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow("yes",scaled_map)
#
# cv2.waitKey(0)
