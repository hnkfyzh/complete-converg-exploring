import cv2
import numpy as np
from PIL import Image
def extract_contours(input_map):
    # Step 1: Dilate the obstacles by 0 and 1 pixel to get border_map1 and border_map2
    kernel = np.ones((5, 5), np.uint8)
    border_map1 = cv2.erode(input_map, kernel, iterations=0)
    border_map2 = cv2.erode(input_map, kernel, iterations=1)

    # Step 2: Define an empty outline map of the same size as the original map
    outline_map = np.zeros_like(input_map)

    # Step 3: XOR the two border maps
    outline_map = cv2.bitwise_xor(border_map1, border_map2)

    # Step 4: Ensure borders are marked as 1 in the outline map
    outline_map[0, :] = 254
    outline_map[-1, :] = 254
    outline_map[:, 0] = 254
    outline_map[:, -1] = 254
    cv2.imshow("image", outline_map)
    cv2.waitKey(0)
    # Step 5: Define a vector to store the result outlines
    result_outline = []

    # Step 6-9: Ray-casting from robot position to find contours
    height, width = outline_map.shape
    center_x, center_y = width // 2, height // 2  # Assuming the robot is at the center of the map

    for angle in range(360):
        x, y = 408, 368
        dx = np.cos(np.radians(angle))
        dy = np.sin(np.radians(angle))

        while 0 <= int(x) < width and 0 <= int(y) < height:
            if outline_map[int(y), int(x)] == 254:
                contour = []
                while 0 <= int(x) < width and 0 <= int(y) < height and outline_map[int(y), int(x)] == 254:
                    contour.append((int(x), int(y)))
                    outline_map[int(y), int(x)] = 0
                    x, y = x + dx, y + dy
                result_outline.append(contour)
                break
            x, y = x + dx, y + dy

    return result_outline

def fill_contours(input_map,result_outline):
    # Step 1: Create an empty destination map of the same size as the original map
    height, width = input_map.shape
    dst_map = np.zeros_like(input_map)

    # Step 2: Mark the contour points in the dst_map
    for contour in result_outline:
        for point in contour:
            dst_map[point[1], point[0]] = 254

    # Step 3-6: Fill the contour area
    stack = []
    for contour in result_outline:
        for point in contour:
            x, y = point
            stack.append((x, y))
            break
        if stack:
            break

    while stack:
        x, y = stack.pop()
        if dst_map[y, x] == 0:
            dst_map[y, x] = 254
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and dst_map[ny, nx] == 0:
                    stack.append((nx, ny))

    return dst_map
# Example usage:
map_img = Image.open('map3--1.pgm')


# 图像处理操作
map_img = np.array(map_img)
map_img = np.rot90(map_img, 3)
# kernel = np.ones((1, 2), np.uint8)
kernel_2 = np.ones((4, 4), np.uint8)
map_img = cv2.erode(map_img, kernel_2, 1)
# input_map = cv2.imread('path_to_input_map.png', cv2.IMREAD_GRAYSCALE)
# map_img = np.where(map_img == 254, 1, map_img)
result_outline = extract_contours(map_img)
print(result_outline)
dst_map = fill_contours(map_img,result_outline)
cv2.imshow("image_dst", dst_map)
cv2.waitKey(0)