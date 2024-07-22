import numpy as np
from collections import deque
from PIL import Image
import cv2


def mark_reachable_points(grid, start):
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4 possible directions (right, down, left, up)
    queue = deque([start])
    visited = set()
    visited.add(start)

    while queue:
        r, c = queue.popleft()
        new_grid[r, c] = 254  # Mark the cell as reachable

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            can_reach = True
            # for car_size_x in range(-2, 3):
            #     for car_size_y in range(-2, 3):
            #         reach_point_x = nr + car_size_x
            #         reach_point_y = nc + car_size_y
            #         if reach_point_x <= 0: reach_point_x = 0
            #         if reach_point_y <= 0: reach_point_y = 0
            #         if reach_point_x >= rows - 1: reach_point_x = rows - 1
            #         if reach_point_y >= cols - 1: reach_point_y = cols - 1
            #         if grid[reach_point_x, reach_point_y] != 254:
            #             can_reach = False

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 254 and (
            nr, nc) not in visited and can_reach != False:
                queue.append((nr, nc))
                visited.add((nr, nc))

    return new_grid


map_img = Image.open('map3--1.pgm')
# 图像处理操作
map_img = np.array(map_img)
map_img = np.rot90(map_img, 3)
# kernel = np.ones((1, 2), np.uint8)
kernel_2 = np.ones((19, 19), np.uint8)
_, map_img = cv2.threshold(map_img, 127, 255, cv2.THRESH_BINARY)
map_img = cv2.erode(map_img, kernel_2, 1)
map_img = map_img[int(map_img.shape[0] / 5):, :]  # 图片裁剪
cv2.imshow("image_dst", map_img)
cv2.waitKey(0)


map_img = np.where(map_img == 255, 254, map_img)
start = (377,1500)  # Starting point (row, col)
new_grid = mark_reachable_points(map_img, start)
cv2.imshow("image_dst", new_grid)
cv2.waitKey(0)

print(new_grid)