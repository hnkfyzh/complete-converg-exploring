from collections import deque

def construct_distance_map(grid, start):
    rows, cols = len(grid), len(grid[0])
    distance_map = [[-2] * cols for _ in range(rows)]  # 初始化距离地图为-2
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 四个方向

    queue = deque([start])
    distance_map[start[0]][start[1]] = 0

    while queue:
        x, y = queue.popleft()
        current_distance = distance_map[x][y]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 254 and (distance_map[nx][ny] == -2 or distance_map[nx][ny] > current_distance + 1):
                    distance_map[nx][ny] = current_distance + 1
                    queue.append((nx, ny))
                elif grid[nx][ny] == 205 and (distance_map[nx][ny] == -2 or distance_map[nx][ny] > current_distance + 11):
                    distance_map[nx][ny] = current_distance + 11
                    queue.append((nx, ny))

    return distance_map

# 示例地图
grid = [
    [254, 254, 254, 0, 254, 254, 254, 0, 254],
    [254, 0, 254, 0, 254, 0, 254, 0, 254],
    [254, 254, 254, 254, 254, 254, 254, 254, 254],
    [0, 0, 0, 254, 0, 0, 0, 254, 0],
    [254, 254, 254, 254, 254, 254, 254, 254, 254],
    [254, 0, 254, 0, 205, 0, 254, 0, 254],
    [254, 254, 254, 254, 205, 254, 254, 254, 254],
    [0, 0, 0, 205, 0, 0, 0, 205, 0],
    [254, 254, 254, 0, 254, 254, 254, 0, 254]
]

start = (4, 4)  # 机器人初始位置

distance_map = construct_distance_map(grid, start)

for row in distance_map:
    print(row)