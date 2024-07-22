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
from scipy.ndimage import zoom
import cv2
import keyboard
from collections import deque
class GridEnv(gym.Env):
    def __init__(self, resolution, num_agents, max_steps,
                map_name,sensor_range,scale_factor,robot_size,
                 agent_0_pos,agent_1_pos,agent_2_pos,
                 use_merge=True,
                 use_same_location=True,
                 use_complete_reward=True,
                 use_multiroom=False,
                 use_time_penalty=False,
                 use_single_reward=False,
                 visualization=False):
        self.finish_exploration = [False,False,False]
        self.finish_exploration_step = [False,False,False]

        self.stop = False

        self.num_agents = num_agents
        self.map_name = map_name

        # map_img = Image.open(map_file)
        # self.gt_map = np.array(map_img)
        # self.inflation_map = obstacle_inflation(self.gt_map, 0.15, 0.05)
        self.scale_factor = scale_factor
        self.robot_size = int(robot_size*self.scale_factor*100/2)+1
        # self.robot_size = 1
        self.resolution = resolution
        self.sensor_range = int(sensor_range*self.scale_factor*100/2)
        # self.sensor_range = 5
        self.agent_0_pos = agent_0_pos
        self.agent_1_pos = agent_1_pos
        self.agent_2_pos = agent_2_pos
        self.pos_traj = []
        self.grid_traj = []
        self.map_per_frame = []
        self.step_time = 0.1
        self.dw_time = 1
        self.minDis2Frontier = 2 * self.resolution
        self.frontiers = []
        self.last_frs = []
        self.path_log = []
        for e in range(self.num_agents):
            self.path_log.append([])
        self.built_map = []


        # self.width = self.gt_map.shape[1]
        # self.height = self.gt_map.shape[0]

        self.resize_width = 64
        self.resize_height = 64

        self.robot_discrete_dir = [i * math.pi for i in range(16)]
        self.agent_view_size = int(sensor_range / self.resolution)
        self.target_ratio = 0.98
        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((num_agents))
        self.agent_ratio_step = np.ones((num_agents)) * max_steps
        self.merge_ratio_step = max_steps
        self.max_steps = max_steps
        # self.total_cell_size = np.sum((self.gt_map != 205).astype(int))
        self.use_same_location = use_same_location
        self.use_complete_reward = use_complete_reward
        self.use_multiroom = use_multiroom
        self.use_time_penalty = use_time_penalty
        self.use_merge = use_merge
        self.use_single_reward = use_single_reward

        # define space
        self.action_space = [gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) for _ in
                             range(self.num_agents)]

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        global_observation_space = {}
        global_observation_space['global_obs'] = gym.spaces.Box(
            low=0, high=255, shape=(4, self.resize_width, self.resize_height), dtype='uint8')

        # global_observation_space['global_merge_goal'] = gym.spaces.Box(
        # low=0, high=255, shape=(2, self.width, self.height), dtype='uint8')

        # global_observation_space['image'] = gym.spaces.Box(
        #     low=0, high=255, shape=(self.resize_width, self.resize_height, 3), dtype='uint8')

        # global_observation_space['vector'] = gym.spaces.Box(
        #     low=-1, high=1, shape=(self.num_agents,), dtype='float')
        global_observation_space['vector'] = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype='float')
        if use_merge:
            global_observation_space['global_merge_obs'] = gym.spaces.Box(
                low=0, high=255, shape=(4, self.resize_width, self.resize_height), dtype='uint8')
            # global_observation_space['global_direction'] = gym.spaces.Box(
            #     low=-1, high=1, shape=(self.num_agents, 4), dtype='float')
        # else:
        #     global_observation_space['global_direction'] = gym.spaces.Box(
        #         low=-1, high=1, shape=(1, 4), dtype='float')
        share_global_observation_space = global_observation_space.copy()
        # share_global_observation_space['gt_map'] = gym.spaces.Box(
        #     low=0, high=255, shape=(1, self.width, self.height), dtype='uint8')

        global_observation_space = gym.spaces.Dict(global_observation_space)
        share_global_observation_space = gym.spaces.Dict(share_global_observation_space)

        self.observation_space = []
        self.share_observation_space = []

        for agent_id in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(share_global_observation_space)

        self.visualization = visualization
        if self.visualization:
            self.window = Window('map')
            # self.window.show(block=False)

        # self.visualize_map = np.zeros((self.width, self.height))
        self.visualize_goal = [[0, 0] for i in range(self.num_agents)]

    def on_space(self):
        print("Space key pressed! program will finish soon.")
        self.stop = True

    def scale_binary_map(self,binary_map, scale_factor):
        # Use zoom function for nearest-neighbor interpolation
        scaled_map = zoom(binary_map, scale_factor, order=0, mode='nearest')

        # Threshold the scaled map to convert it back to a binary map
        scaled_map = (scaled_map > 0.5).astype(int)

        return scaled_map

#图像处理，可到达点像素为254，不可到达点像素为0
    def mark_reachable_points(self,grid, start):
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
                # for car_size_x in range(-1, 2):
                #     for car_size_y in range(-1, 2):
                #         if grid[nr + car_size_x, nc + car_size_y] != 254:
                #             can_reach = False
                #             break
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 254 and (nr, nc) not in visited and can_reach != False:
                    queue.append((nr, nc))

                    visited.add((nr, nc))

        return new_grid
    def reset_for_traditional(self):
        # 1. read from blueprints files randomly
        # map_file = random.choice(os.listdir('/home/nics/workspace/blueprints'))
        # map_img = Image.open(os.path.join('/home/nics/workspace/blueprints', map_file))
        global map_img
        map_img = Image.open(self.map_name)
        self.gt_map = np.array(map_img)  # 图

        #图像处理操作
        map_img = np.array(map_img)
        # height = int(map_img.shape[0]*0.2)
        # width = int(map_img.shape[1]*0.2)
        # map_img = cv2.resize(map_img,(1886,1331),interpolation=cv2.INTER_NEAREST) #resize成(1886，1331)
        # # 初始化全局变量
        # global ref_point
        # ref_point = []
        # global cropping
        # cropping = False
        #
        # def click_and_crop(event, x, y, flags, param):
        #     global ref_point, cropping, map_img
        #
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         ref_point = [(x, y)]
        #         cropping = True
        #
        #     elif event == cv2.EVENT_MOUSEMOVE:
        #         if cropping:
        #             temp_image = image.copy()
        #             cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
        #             cv2.imshow("image", temp_image)
        #
        #     elif event == cv2.EVENT_LBUTTONUP:
        #         ref_point.append((x, y))
        #         cropping = False
        #
        #         cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        #         cv2.imshow("image", image)
        #
        # # 加载图像并设置窗口
        # image = map_img
        # clone = image.copy()
        # cv2.namedWindow("image")
        # cv2.setMouseCallback("image", click_and_crop)
        #
        # while True:
        #     cv2.imshow("image", image)
        #     key = cv2.waitKey(1) & 0xFF
        #
        #     if key == ord("r"):
        #         image = clone.copy()
        #
        #     elif key == ord("c"):
        #         if len(ref_point) == 2:
        #             roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        #             cv2.imshow("ROI", roi)
        #             cv2.imwrite('cropped_image.jpg', roi)
        #             map_img = roi
        #             cv2.waitKey(0)
        #         break
        #
        #     elif key == 27:
        #         break
        #
        # cv2.destroyAllWindows()

        # map_img = np.rot90(map_img,3)
        # map_img = map_img[int(map_img.shape[0] / 5):, :]  # 图片裁剪

        # kernel_1 = np.ones((15, 15), np.uint8)
        #
        # # _, self.show_observation_image = cv2.threshold(self.show_observation_image, 127, 255, cv2.THRESH_BINARY)
        # self.show_observation_image = cv2.erode(map_img, kernel_1, 1)
        # self.show_observation_image = self.scale_binary_map(self.show_observation_image, self.scale_factor)
        # self.show_observation_image = np.where(self.show_observation_image == 1, 254, self.show_observation_image)

        # self.show_observation_image = self.show_observation_image[int(self.show_observation_image.shape[0] / 5):, :]
        # kernel = np.ones((1, 2), np.uint8)
        kernel_2 = np.ones((15, 15), np.uint8)
        # scaled_map = cv2.dilate(map_img, kernel, 1)
        print("please waiting")
        #在腐蚀之前一定要转为二值图像
        _, scaled_map = cv2.threshold(map_img, 127, 255, cv2.THRESH_BINARY)
        scaled_map = cv2.erode(scaled_map, kernel_2, 1)
        scaled_map = np.where(scaled_map == 255, 254, scaled_map)
        # scaled_map = scaled_map[int(scaled_map.shape[0] / 5):, :]  # 图片裁剪

        # scaled_map = self.scale_binary_map(scaled_map, self.scale_factor)
        # scaled_map = np.where(scaled_map == 1, 254, scaled_map)
        # scaled_map = scaled_map.astype(np.uint8)
        scaled_map = cv2.copyMakeBorder(scaled_map, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # scaled_map = cv2.line(scaled_map, (0, 18), (198, 18), 0, 1)
        self.gt_map = np.array(scaled_map)

        # Scale factor for halving the size

        # self.window.show_img(self.gt_map)
        # import pdb; pdb.set_trace()
        self.inflation_map = obstacle_inflation(self.gt_map, 0.15, 0.05)  # 膨胀操作
        # self.window.show_img(self.inflation_map)
        self.width = self.gt_map.shape[0]
        self.height = self.gt_map.shape[1]
        self.total_cell_size = np.sum((self.gt_map != 205).astype(int))  # 为什么是灰度205，未知区域？
        self.visualize_map = np.zeros((self.width, self.height))  # 初始是全是0

        self.num_step = 0
        obs = []
        self.built_map = []
        # reset robot pos and dir
        # self.agent_pos = [self.continuous_to_discrete([-8, 8.2]), self.continuous_to_discrete([8, -8])]
        # self.agent_pos = [self.continuous_to_discrete([-8, 8])] # 7.2
        # self.agent_dir = [0]
        self.agent_pos = []  # 7.2
        self.agent_dir = []  # 这个是朝向


        a = []
        b = []

        scaled_map_uint8 = scaled_map.astype(np.uint8)   #最原始的地图
        # self.show_observation_image = scaled_map_uint8


        scaled_map_uint8 = cv2.resize(scaled_map_uint8,(1886,1331),interpolation=cv2.INTER_LINEAR) #resize成(1886，1331)

        # scaled_map_uint8 = cv2.resize(scaled_map_uint8,(500,200),interpolation=cv2.INTER_LINEAR)
        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                xy = "[%d,%d]" % (x, y)
                a.append(x)
                b.append(y)
                cv2.circle(scaled_map_uint8, (x, y), 3, (255, 0, 0), thickness=-1)
                cv2.putText(scaled_map_uint8, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)  # 如果不想在图片上显示坐标可以注释该行

                cv2.imshow("image", scaled_map_uint8)
                print("[{},{}]".format(a[-1], b[-1]))  # 终端中输出坐标
                # self.agent_pos.append([int(b[-1]*self.width*self.scale_factor/1331), int(a[-1]*self.height*self.scale_factor/1886)])
                self.agent_pos.append([int(b[-1] * self.width  / 1331),
                                       int(a[-1] * self.height / 1886)])


        cv2.namedWindow("image")
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", scaled_map_uint8) #这一步是适配（1886，1331）
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        #下面这行代码是依据机器人所在位置，找到它所有能到达的位置
        new_grid = self.mark_reachable_points(self.gt_map, (self.agent_pos[0][0],self.agent_pos[0][1]))
        self.gt_map = np.array(new_grid)
        # scaled_map_uint8 = cv2.resize(new_grid, (1886, 1331),
        #                               interpolation=cv2.INTER_LINEAR)  # resize成(1886，1331)
        # cv2.imshow("image5", scaled_map_uint8)
        # cv2.waitKey(0)
        # scaled_map_uint8 = cv2.resize(new_grid, (1886, 1331),
        #                               interpolation=cv2.INTER_LINEAR)  # resize成(1886，1331)
        # self.agent_pos.append([int(self.agent_0_pos[0]*self.scale_factor), int(self.agent_0_pos[1]*self.scale_factor)])
        # self.agent_pos.append([int(self.agent_1_pos[0]*self.scale_factor), int(self.agent_1_pos[1]*self.scale_factor)])
        # self.agent_pos.append([int(self.agent_2_pos[0]*self.scale_factor), int(self.agent_2_pos[1]*self.scale_factor)])
        self.gt_map = self.scale_binary_map(self.gt_map, self.scale_factor)
        self.gt_map = np.where(self.gt_map == 1, 254, self.gt_map)
        self.gt_map = self.gt_map.astype(int)
        self.width = self.gt_map.shape[0]
        self.height = self.gt_map.shape[1]

        # scaled_map_uint8 = cv2.resize(self.gt_map, (1886, 1331), interpolation=cv2.INTER_LINEAR) #适配场景
        # cv2.imshow("image5", scaled_map_uint8)
        # cv2.waitKey(0)

        #后面这里的位置是放缩过的
        self.agent_pos[0][0] = int(self.agent_pos[0][0] * self.scale_factor)
        self.agent_pos[0][1] = int(self.agent_pos[0][1] * self.scale_factor)
        self.agent_pos[1][0] = int(self.agent_pos[1][0] * self.scale_factor)
        self.agent_pos[1][1] = int(self.agent_pos[1][1] * self.scale_factor)
        self.agent_pos[2][0] = int(self.agent_pos[2][0] * self.scale_factor)
        self.agent_pos[2][1] = int(self.agent_pos[2][1] * self.scale_factor)
        self.agent_dir.append(random.randint(0, 3))
        self.agent_dir.append(random.randint(0, 3))
        self.agent_dir.append(random.randint(0, 3))

        # self.agent_pos.append([int(320*self.scale_factor), int(330*self.scale_factor)])
        # self.agent_dir.append(random.randint(0, 3))
        # self.agent_pos.append([int(320*self.scale_factor), int(1800*self.scale_factor)])
        # self.agent_dir.append(random.randint(0, 3))
        # self.agent_pos.append([int(625*self.scale_factor), int(330*self.scale_factor)])
        # self.agent_dir.append(random.randint(0, 3))


        # init local map
        self.explored_each_map = []#每一个机器人探索到的局部地图
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []

        for i in range(self.num_agents):
            self.explored_each_map.append(np.zeros((self.width, self.height)))  # 用于追踪每个代理所探索的区域
            self.obstacle_each_map.append(np.zeros((self.width, self.height)))  # 用于记录每个代理所观察到的障碍物区域
            self.previous_explored_each_map.append(np.zeros((self.width, self.height)))  # 记录之前每个代理探索的区域

        for i in range(self.num_agents):

            # unknown: 205   free: 254   occupied: 0 该机器人这一步所探索到的地图
            # map_this_frame,_,_ = self.optimized_build_map_with_direction(self.agent_pos[i],self.agent_dir[i],self.gt_map,self.sensor_range)
            map_this_frame, _, _ = self.optimized_build_map_mine(self.agent_pos[i], self.agent_dir[i],self.gt_map, self.sensor_range)
            self.built_map.append(map_this_frame)  # 加入到该机器人探索的整体的地图当中
            obs.append(map_this_frame)  #
            current_agent_pos.append(self.agent_pos[i])
            self.explored_each_map[i] = (map_this_frame != 205).astype(int)  # 不是205的是探索过的，然后是True或者false
            self.obstacle_each_map[i] = (map_this_frame == 0).astype(int)  # 这两个难道不会重吗

        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))
        self.previous_all_map = np.zeros((self.width, self.height))
        for i in range(self.num_agents):
            explored_all_map += self.explored_each_map[i]
            obstacle_all_map += self.obstacle_each_map[i]
        explored_all_map = (explored_all_map > 0).astype(int)
        obstacle_all_map = (obstacle_all_map > 0).astype(int)
        #后面解释的很清楚
        # if we have both explored map and obstacle map, we can merge them to get complete map
        # obstacle: 2   free: 1   unknown: 0
        temp = explored_all_map + obstacle_all_map  # 地图， 2表示障碍，1表示free，0 表示未探索区域
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205

        info = {}
        info['explored_all_map'] = np.array(explored_all_map)
        info['current_agent_pos'] = np.array(current_agent_pos)
        info['explored_each_map'] = np.array(self.explored_each_map)
        info['obstacle_all_map'] = np.array(obstacle_all_map)
        info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        info['agent_direction'] = np.array(self.agent_dir)
        # info['agent_local_map'] = self.agent_local_map

        info['merge_explored_ratio'] = self.merge_ratio
        info['merge_explored_reward'] = self.merge_reward
        info['agent_explored_reward'] = self.agent_reward
        info['merge_ratio_step'] = self.merge_ratio_step

        for i in range(self.num_agents):
            info["agent{}_ratio_step".format(i)] = self.agent_ratio_step[i]

        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_reward = np.zeros((self.num_agents))
        self.agent_ratio_step = np.ones((self.num_agents)) * self.max_steps
        self.merge_ratio_step = self.max_steps

        obs = np.array(obs)
        if self.visualization:
            # self.window.show_img(self.built_map[0])
            self.window.show_img(self.complete_map)
        return obs, info

    def detect_frontiers(self, explored_map):
        '''
        detect frontiers from current built map
        '''
        obstacles = []
        frontiers = []
        height = explored_map.shape[0]
        width = explored_map.shape[1]
        for i in range(2, height - 2):
            for j in range(2, width - 2):
                if explored_map[i][j] == 2:
                    obstacles.append([i, j])
                elif explored_map[i][j] == 0:
                    numFree = 0
                    temp1 = 0
                    if explored_map[i + 1][j] == 1:
                        temp1 += 1 if explored_map[i + 2][j] == 1 else 0
                        temp1 += 1 if explored_map[i + 1][j + 1] == 1 else 0
                        temp1 += 1 if explored_map[i + 1][j - 1] == 1 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i][j + 1] == 1:
                        temp1 += 1 if explored_map[i][j + 2] == 1 else 0
                        temp1 += 1 if explored_map[i + 1][j + 1] == 1 else 0
                        temp1 += 1 if explored_map[i - 1][j + 1] == 1 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i - 1][j] == 1:
                        temp1 += 1 if explored_map[i - 1][j + 1] == 1 else 0
                        temp1 += 1 if explored_map[i - 1][j - 1] == 1 else 0
                        temp1 += 1 if explored_map[i - 2][j] == 1 else 0
                        numFree += (temp1 > 0)
                    if explored_map[i][j - 1] == 1:
                        temp1 += 1 if explored_map[i][j - 2] == 1 else 0
                        temp1 += 1 if explored_map[i + 1][j - 1] == 1 else 0
                        temp1 += 1 if explored_map[i - 1][j - 1] == 1 else 0
                        numFree += (temp1 > 0)
                    if numFree > 0:
                        frontiers.append([i, j])
        return frontiers, obstacles


    def optimized_build_map_mine(self, pos, direction,gt_map, sensor_range):
        height = gt_map.shape[0]
        width = gt_map.shape[1]
        x_min, y_min = max(0, pos[0] - sensor_range-1), max(0, pos[1] - sensor_range-1)
        x_max, y_max = min(pos[0] + sensor_range, height), min(pos[1] + sensor_range, width)
        scale_map = np.zeros(gt_map.shape)
        scale_map[:, :] = 205
        for i in range(-sensor_range+1,sensor_range):
            for j in range(-sensor_range+1,sensor_range):
                x = pos[0] + i
                y = pos[1] + j
                if x <= 0:
                    x = 0
                if x >= height - 1:
                    x = height - 1
                if y <= 0:
                    y = 0
                if y >= width - 1:
                    y = width - 1
                scale_map[x,y] = gt_map[x,y]


        return scale_map,(x_min, x_max), (y_min, y_max)

    def optimized_build_map_with_direction(self, pos, direction, gt_map, sensor_range,robot_num):
        height = gt_map.shape[0]
        width = gt_map.shape[1]
        x_min, y_min = max(0, pos[0] - sensor_range-1), max(0, pos[1] - sensor_range-1)
        x_max, y_max = min(pos[0] + sensor_range, height), min(pos[1] + sensor_range, width)
        scale_map = np.zeros(gt_map.shape)
        scale_map[:, :] = 205
        for i in range(-self.robot_size, self.robot_size+1):
            for j in range(-self.robot_size, self.robot_size+1):
                gt_map[pos[0],pos[1]] = robot_num+1 #走过的路不能再走了

        for i in range(-sensor_range+1,sensor_range):
            for j in range(-sensor_range+1,sensor_range):
                if direction == 1: #下
                    if i >= -j and i>= j and i>=0:
                        x = pos[0] + i
                        y = pos[1] + j
                        if x <= 0:
                            x = 0
                        if x >= height - 1:
                            x = height - 1
                        if y <= 0:
                            y = 0
                        if y >= width - 1:
                            y = width - 1
                        scale_map[x,y] = gt_map[x,y]

                if direction == 0:  #上
                    if -i >= j and -i>= -j and i<=0:
                        x = pos[0] + i
                        y = pos[1] + j
                        if x <= 0:
                            x = 0
                        if x >= height - 1:
                            x = height - 1
                        if y <= 0:
                            y = 0
                        if y >= width - 1:
                            y = width - 1
                        scale_map[x,y] = gt_map[x,y]

                if direction == 2:  #左
                    if -j >= -i and -j>= i and j<=0 :
                        x = pos[0] + i
                        y = pos[1] + j
                        if x <= 0:
                            x = 0
                        if x >= height - 1:
                            x = height - 1
                        if y <= 0:
                            y = 0
                        if y >= width - 1:
                            y = width - 1
                        scale_map[x,y] = gt_map[x,y]

                if direction == 3:  #右
                    if j >= -i and j>= i and j>=0:
                        x = pos[0] + i
                        y = pos[1] + j
                        if x <= 0:
                            x = 0
                        if x >= height - 1:
                            x = height - 1
                        if y <= 0:
                            y = 0
                        if y >= width - 1:
                            y = width - 1
                        scale_map[x,y] = gt_map[x,y]



        return scale_map,(x_min, x_max), (y_min, y_max)


    def merge_two_map(self, map1, map2, x, y):
        '''
        merge two map into one map
        should be accelerated
        '''
        test_map = map1 + map2
        merge_map = copy.deepcopy(map1)
        for i in range(x[0], x[1]):
            for j in range(y[0], y[1]):
                if test_map[i][j] == 0 or test_map[i][j] == 205 or test_map[i][j] == 254:
                    merge_map[i][j] = 0
                elif test_map[i][j] == 410:
                    merge_map[i][j] = 205
                elif test_map[i][j] == 459 or test_map[i][j] == 508:
                    merge_map[i][j] = 254
        return merge_map

    def continuous_to_discrete(self, pos):
        idx_x = int(pos[0] / self.resolution) + int(self.gt_map.shape[0] / 2)
        idx_y = int(pos[1] / self.resolution) + int(self.gt_map.shape[1] / 2)
        return [idx_x, idx_y]

    def discrete_to_continuous(self, grid_idx):
        pos_x = (grid_idx[0] - self.gt_map.shape[0] / 2) * self.resolution
        pos_y = (grid_idx[1] - self.gt_map.shape[1] / 2) * self.resolution
        return [pos_x, pos_y]

    def naive_local_planner(self, global_plan):
        '''
        Naive local planner
        always move along the global path
        '''
        pose = []
        # add orn to global path
        for idx, pos in enumerate(global_plan):
            if idx == 0:
                pose.append(self.calculate_pose(c_pos=pos, n_pos=global_plan[idx + 1]))
            elif idx == len(global_plan) - 1:
                pose.append(self.calculate_pose(c_pos=pos, p_pos=global_plan[idx - 1]))
            else:
                pose.append(self.calculate_pose(c_pos=pos, p_pos=global_plan[idx - 1], n_pos=global_plan[idx + 1]))
        return pose

    def calculate_pose(self, p_pos=None, c_pos=None, n_pos=None):
        '''
        For naive local planner only
        p_pos: previous robot's position
        c_pos: current robot's position
        n_pos: next robot's position
        '''
        # n_pos - c_pos
        start_pos2orn = {(-1, -1): 3 * math.pi / 4, (-1, 0): math.pi / 2, (-1, 1): math.pi / 4, (0, 1): 0,
                         (1, 1): 7 * math.pi / 4, (1, 0): 3 * math.pi / 2, (1, -1): 5 * math.pi / 4, (0, -1): math.pi}
        if not p_pos:
            return [c_pos, start_pos2orn[tuple((np.array(n_pos) - np.array(c_pos)).tolist())]]
        # p_pos - c_pos
        end_pos2orn = {(-1, -1): 7 * math.pi / 4, (-1, 0): 3 * math.pi / 2, (-1, 1): 5 * math.pi / 4, (0, 1): math.pi,
                       (1, 1): 3 * math.pi / 4, (1, 0): math.pi / 2, (1, -1): math.pi / 4, (0, -1): 0}
        if not n_pos:
            return [c_pos, end_pos2orn[tuple((np.array(p_pos) - np.array(c_pos)).tolist())]]

        # tuple (p_pos - c_pos, n_pos - c_pos)
        mid_end_pos2orn = {(-1, -1, -1, 1): 0, (-1, -1, 0, 1): 15 * math.pi / 8, (-1, -1, 1, 1): 7 * math.pi / 4,
                           (-1, -1, 1, 0): 13 * math.pi / 8, (-1, -1, 1, -1): 3 * math.pi / 2,
                           (-1, 0, 0, 1): 7 * math.pi / 4, (-1, 0, 1, 1): 13 * math.pi / 8,
                           (-1, 0, 1, 0): 3 * math.pi / 2, (-1, 0, 1, -1): 11 * math.pi / 8,
                           (-1, 0, 0, -1): 5 * math.pi / 4,
                           (-1, 1, 1, 1): 3 * math.pi / 2, (-1, 1, 1, 0): 11 * math.pi / 8,
                           (-1, 1, 1, -1): 5 * math.pi / 4, (-1, 1, 0, -1): 9 * math.pi / 8, (-1, 1, -1, -1): math.pi,
                           (0, 1, 1, 0): 5 * math.pi / 4, (0, 1, 1, -1): 9 * math.pi / 8, (0, 1, 0, -1): math.pi,
                           (0, 1, -1, -1): 7 * math.pi / 8, (0, 1, -1, 0): 3 * math.pi / 4,
                           (1, 1, 1, -1): math.pi, (1, 1, 0, -1): 7 * math.pi / 8, (1, 1, -1, -1): 3 * math.pi / 4,
                           (1, 1, -1, 0): 5 * math.pi / 8, (1, 1, -1, 1): math.pi / 2,
                           (1, 0, 0, -1): 3 * math.pi / 4, (1, 0, -1, -1): 5 * math.pi / 8, (1, 0, -1, 0): math.pi / 2,
                           (1, 0, -1, 1): 3 * math.pi / 8, (1, 0, 0, 1): math.pi / 4,
                           (1, -1, -1, -1): math.pi / 2, (1, -1, -1, 0): 3 * math.pi / 8, (1, -1, -1, 1): math.pi / 4,
                           (1, -1, 0, 1): math.pi / 8, (1, -1, 1, 1): 0,
                           (0, -1, -1, 0): math.pi / 4, (0, -1, -1, 1): math.pi / 8, (0, -1, 0, 1): 0,
                           (0, -1, 1, 1): 15 * math.pi / 8, (0, -1, 1, 0): 7 * math.pi / 4}
        return [c_pos, mid_end_pos2orn[
            tuple(np.concatenate([np.array(p_pos) - np.array(c_pos), np.array(n_pos) - np.array(c_pos)]).tolist())]]

    def ObstacleCostFunction(self, trajectory):
        for each in trajectory:
            if self.inflation_map[each[0], each[1]] == 0:
                return True
            else:
                pass
        return False

    def Astar_global_planner(self, start, goal,robot_num):

        # start_pos = self.continuous_to_discrete(start)
        # goal_pos = self.continuous_to_discrete(goal)
        astar = AStar(tuple(start), tuple(goal), self.gt_map, self.robot_size,"euclidean")
        # plot = plotting.Plotting(s_start, s_goal)
        astar.robot_num = robot_num
        path, visited = astar.searching()

        # vis_map = self.plot_path(path)
        # img = Image.fromarray(vis_map.astype('uint8'))
        # img.show()
        # import pdb; pdb.set_trace()
        if path != 0:
            return list(reversed(path))
        else:
            return 0

    def point_to_path_min_distance(self, point, path):
        dis = []
        for each in path:
            d_each = self.discrete_to_continuous(each)
            dis.append(math.hypot(point[0] - d_each[0], point[1] - d_each[1]))
        return min(dis), dis.index(min(dis))

    def informationRectangleGain(self, mapData, point, r):
        infoGainValue = 0
        r_region = int(r / self.resolution)
        point = self.continuous_to_discrete(point)
        # if point[0]+r_region < mapData.shape[0] and point[1]+r_region < mapData.shape[1]:
        #     for i in range(point[0]-r_region, point[0]+r_region+1):
        #         for j in range(point[1]-r_region, point[1]+r_region+1):
        #             if mapData[i][j] == 205:
        #                 infoGainValue += 1
        #             elif mapData[i][j] == 0:
        #                 infoGainValue -= 1
        # else:
        for i in range(point[0] - r_region, min(point[0] + r_region + 1, mapData.shape[0])):
            for j in range(point[1] - r_region, min(point[1] + r_region + 1, mapData.shape[1])):
                if mapData[i][j] == 205:
                    infoGainValue += 1
                elif mapData[i][j] == 0:
                    infoGainValue -= 1
        tempResult = infoGainValue * math.pow(self.resolution, 2)
        return tempResult

    def dismapConstruction_start_target(self, curr, map):
        curr_iter = []
        next_iter = []

        iter = 1
        LARGEST_MAP_DISTANCE = 500 * 1000
        curr_iter.append(curr)

        dismap_backup = copy.deepcopy(map)
        dismap_ = copy.deepcopy(map)
        # dismap_: obstacle -2  unknown -1 free 0
        # built_map: obstacle 0 unknown 205 free 254
        for i in range(dismap_.shape[0]):
            for j in range(dismap_.shape[1]):
                if dismap_backup[i][j] == 0:
                    dismap_[i][j] = -2
                if dismap_backup[i][j] == 205:
                    dismap_[i][j] = -1
                if dismap_backup[i][j] == 254:
                    dismap_[i][j] = 0
        dismap_[curr[0], curr[1]] = -500

        while (len(curr_iter)) > 0:
            if iter > LARGEST_MAP_DISTANCE:
                print("distance exceeds MAXIMUM SETUP")
                return
            for i in range(len(curr_iter)):
                if dismap_[curr_iter[i][0] + 1, curr_iter[i][1]] == 0:
                    dismap_[curr_iter[i][0] + 1, curr_iter[i][1]] = iter
                    next_iter.append([curr_iter[i][0] + 1, curr_iter[i][1]])
                if dismap_[curr_iter[i][0], curr_iter[i][1] + 1] == 0:
                    dismap_[curr_iter[i][0], curr_iter[i][1] + 1] = iter
                    next_iter.append([curr_iter[i][0], curr_iter[i][1] + 1])
                if dismap_[curr_iter[i][0] - 1, curr_iter[i][1]] == 0:
                    dismap_[curr_iter[i][0] - 1, curr_iter[i][1]] = iter
                    next_iter.append([curr_iter[i][0] - 1, curr_iter[i][1]])
                if dismap_[curr_iter[i][0], curr_iter[i][1] - 1] == 0:
                    dismap_[curr_iter[i][0], curr_iter[i][1] - 1] = iter
                    next_iter.append([curr_iter[i][0], curr_iter[i][1] - 1])
            curr_iter = copy.deepcopy(next_iter)
            next_iter = []
            iter += 1

        dismap_[curr[0], curr[1]] = 0

        # window = Window('path')
        # window.show(block=False)
        # window.show_img(dismap_)
        # import pdb; pdb.set_trace()

        return dismap_

    def plot_map_with_path(self):
        vis = copy.deepcopy(self.complete_map)
        for e in range(self.num_agents):
            for pose in self.path_log[e]:
                if e == 0:
                    vis[pose[0], pose[1]] = 64
                elif e == 1:
                    vis[pose[0], pose[1]] = 128
                elif e == 2:
                    vis[pose[0], pose[1]] = 192
        self.window.show_img(vis)

    def frontiers_detection_for_cost(self, map):
        '''
        detect frontiers from current built map
        '''
        obstacles = []
        frontiers = []
        height = map.shape[0]
        width = map.shape[1]
        for i in range(2, height - 2):
            for j in range(2, width - 2):
                if map[i][j] == 0 :
                    obstacles.append([i, j])
                elif map[i][j] == 205:
                    numFree = 0
                    temp1 = 0
                    if map[i + 1][j] == 254:
                        temp1 += 1 if map[i + 2][j] == 254 else 0
                        temp1 += 1 if map[i + 1][j + 1] == 254 else 0
                        temp1 += 1 if map[i + 1][j - 1] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i][j + 1] == 254:
                        temp1 += 1 if map[i][j + 2] == 254 else 0
                        temp1 += 1 if map[i + 1][j + 1] == 254 else 0
                        temp1 += 1 if map[i - 1][j + 1] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i - 1][j] == 254:
                        temp1 += 1 if map[i - 1][j + 1] == 254 else 0
                        temp1 += 1 if map[i - 1][j - 1] == 254 else 0
                        temp1 += 1 if map[i - 2][j] == 254 else 0
                        numFree += (temp1 > 0)
                    if map[i][j - 1] == 254:
                        temp1 += 1 if map[i][j - 2] == 254 else 0
                        temp1 += 1 if map[i + 1][j - 1] == 254 else 0
                        temp1 += 1 if map[i - 1][j - 1] == 254 else 0
                        numFree += (temp1 > 0)
                    if numFree > 0:
                        frontiers.append([i, j])
        return frontiers, obstacles

    def get_goal_for_cost(self):
        map_goal = [[],[],[]]
        map_goal_dis = [[],[],[]]
        global_plan_accumulate = [[],[],[]]
        while self.finish_exploration_step != [True,True,True] and self.stop == False: #直到全部探索完
            e = self.finish_exploration_step.index(False)
        # for e in range(self.num_agents):
            # obstacle: 2  unknown: 0   free: 1
            if self.finish_exploration[e] == False:
                frs, _ = self.frontiers_detection_for_cost(self.complete_map)#边缘点检测
                # cluster targets into different groups and find the center of each group.

                # 上一次的去掉
                # tuple_last_set_frs = [tuple(sublist) for sublist in self.last_frs]
                # last_set_frs = set(tuple_last_set_frs)
                # tuple_frs = [tuple(sublist) for sublist in frs]
                # set_frs = set(tuple_frs)
                # frs = [list(tpl) for tpl in set_frs - last_set_frs]  #上一次的去掉

                target_process = copy.deepcopy(frs)
                cluster_center = []
                infoGain_cluster = []

                #聚类操作
                while (len(target_process) > 0):
                    target_cluster = []
                    target_cluster.append(target_process.pop())  #target_process放出来一个
                    target_cluster_first = target_cluster[0]

                    condition = True
                    while (condition):
                        condition = False
                        size_target_process = len(target_process)
                        for i in reversed(range(size_target_process)):
                            for j in range(len(target_cluster)):
                                delta_x = abs(target_process[i][0] - target_cluster[j][0])
                                delta_y = abs(target_process[i][1] - target_cluster[j][1])

                                delta_x_first = abs(target_cluster_first[0] - target_process[i][0])
                                delta_y_first = abs(target_cluster_first[1] - target_process[i][1])
                                dis_first = delta_x_first + delta_y_first

                                dis = delta_x+delta_y
                                # if dis_first < 15:
                                # if dis < 2 and dis_first < 7: #如果计算出的距离小于2，说明两个点足够近，可以合并
                                if dis < 2: #如果计算出的距离小于2，说明两个点足够近，可以合并
                                    target_cluster.append(target_process[i])
                                    del target_process[i]
                                    condition = True
                                    break

                    center_ = [0, 0]
                    num_ = len(target_cluster)
                    for i in range(num_):
                        center_[0] += target_cluster[i][0]
                        center_[1] += target_cluster[i][1]

                    center_float = [float(center_[0]) / float(num_), float(center_[1]) / float(num_)]
                    min_dis_ = 100.0
                    min_idx_ = 10000
                    for i in range(num_): #target_cluster的数量
                        temp_dis_ = abs(center_float[0] - float(target_cluster[i][0])) + abs(
                            center_float[1] - float(target_cluster[i][1]))
                        if temp_dis_ < min_dis_:
                            min_dis_ = temp_dis_
                            min_idx_ = i

                    cluster_center.append([target_cluster[min_idx_][0], target_cluster[min_idx_][1]])
                    infoGain_cluster.append(num_)


                # curr_dismap = self.dismapConstruction_start_target(self.agent_pos[e], self.built_map[e])
                curr_dismap = self.dismapConstruction_start_target(self.agent_pos[e], self.gt_map)#构建的距离地图
                Dis2Frs = []
                free_cluster_center = []
                for i in range(len(cluster_center)):
                    # find the nearest free grid
                    #经过这样一个循环还能保持通行
                    for a in range(-(self.sensor_range-int(self.robot_size*1.5)),self.sensor_range-int(self.robot_size*1.5)+1,int(self.robot_size*1.5)):   #从0开始
                        for b in range(-(self.sensor_range-int(self.robot_size*1.5)),self.sensor_range-int(self.robot_size*1.5)+1,int(self.robot_size*1.5)):#周围区域检测可到达点
                    # for a in range(-3,4):  # 从0开始
                    #     for b in range(-3,4):  # 周围区域检测可到达点
                            can_reach = True
                            reach_point_x = cluster_center[i][0] + a
                            reach_point_y = cluster_center[i][1] + b
                            boundary_x = self.gt_map.shape[0]
                            boundary_y = self.gt_map.shape[1]
                            if reach_point_x <= 0: reach_point_x = 0
                            if reach_point_y <= 0: reach_point_y = 0
                            if reach_point_x >= boundary_x-1: reach_point_x = boundary_x-1
                            if reach_point_y >= boundary_y-1: reach_point_y = boundary_y-1
                            #碰撞检测
                            # for car_size_x in range(-self.robot_size,self.robot_size+1):
                            #     for car_size_y in range(-self.robot_size,self.robot_size+1):
                            for car_size_x in range(1):
                                for car_size_y in range(1):
                                    # if (self.gt_map[reach_point_x + car_size_x, reach_point_y + car_size_y] != 254
                                    #         or curr_dismap[reach_point_x, reach_point_y]>=120):
                                    if self.gt_map[reach_point_x + car_size_x, reach_point_y + car_size_y] != 254:
                                        can_reach = False
                                        break
                                if can_reach == False:
                                    break
                            if can_reach == True:
                                Dis2Frs.append(curr_dismap[reach_point_x, reach_point_y])
                                free_cluster_center.append([reach_point_x, reach_point_y])

                find_suitable_point = False
                find_different_point = False
                while(not find_suitable_point) and self.stop == False:#一个循环，使机器人都找到合适的目标点，否则就finish
                    if Dis2Frs != []:
                        min_Dis2Frs = min(Dis2Frs)
                        # map_goal_dis = [[], [], []]
                        index = Dis2Frs.index(min_Dis2Frs)
                        goal = free_cluster_center[index]
                        global_plan = self.Astar_global_planner(self.agent_pos[e], np.array(goal),e)


                        # if goal in map_goal:#防止混水摸鱼现象,其它机器人当作目标的点不再作为另外一个机器人的目标点
                        # # if goal_similar == True:  # 防止混水摸鱼现象,其它机器人当作目标的点不再作为另外一个机器人的目标点
                        #     last_goal_index = map_goal.index(goal)
                        #     if min_Dis2Frs < map_goal_dis[last_goal_index] and global_plan != 0 and goal[0] != self.agent_pos[e][0] and goal[1] != self.agent_pos[e][1]:#当前目标点的距离小于上一个机器人目标点的距离
                        #         map_goal[e] = goal
                        #         map_goal_dis[e] = min_Dis2Frs
                        #         global_plan_accumulate[e] = global_plan
                        #         find_suitable_point = True
                        #         self.finish_exploration_step[e] = True
                        #         self.finish_exploration_step[last_goal_index] = False
                        #
                        #     else:
                        #         free_cluster_center.pop(index)  #这一个聚合点不再作为我们的目标点
                        #         Dis2Frs.pop(index)
                        #
                        #         target_process_free_cluster_center = copy.deepcopy(free_cluster_center)
                        #         # 聚类操作
                        #         target_free_cluster = []
                        #         target_free_cluster.append(goal)  # target_process放出来一个
                        #
                        #
                        #         condition = True
                        #         while (condition):
                        #             condition = False
                        #             size_target_process_free_cluster_center = len(target_process_free_cluster_center)
                        #             for i in reversed(range(size_target_process_free_cluster_center)):
                        #                 for j in range(len(target_free_cluster)):
                        #                     delta_x = abs(target_process_free_cluster_center[i][0] - target_free_cluster[j][0])
                        #                     delta_y = abs(target_process_free_cluster_center[i][1] - target_free_cluster[j][1])
                        #
                        #                     dis = delta_x + delta_y
                        #                         # if dis < 2 and (delta_x_first == 0 or delta_y_first == 0): #如果计算出的距离小于2，说明两个点足够近，可以合并
                        #                     if dis < 20:  # 目标点附近的点都不作为当前机器人的目标点
                        #                         index_surround = free_cluster_center.index(target_process_free_cluster_center[i])
                        #                         free_cluster_center.pop(index_surround)  # 这一个聚合点不再作为我们的目标点
                        #                         Dis2Frs.pop(index_surround)
                        #                         target_free_cluster.append(target_process_free_cluster_center[i])
                        #                         del target_process_free_cluster_center[i]
                        #                         condition = True
                        #                         break

                        if global_plan != 0 and goal[0] != self.agent_pos[e][0] and goal[1] != self.agent_pos[e][1]:
                            # 目标点可以到达
                            # if goal not in map_goal:
                            # 保证和初始点不同
                            # Dis2Frs是聚类中心到free区域的距离，如果有两个距离一样的呢？
                            map_goal[e]=goal
                            map_goal_dis[e] = min_Dis2Frs
                            global_plan_accumulate[e]=global_plan
                            find_suitable_point = True
                            self.finish_exploration_step[e] = True

                        elif len(free_cluster_center)!=0: #还有其它的聚合点可选择
                            free_cluster_center.pop(index)  #这一个聚合点不再作为我们的目标点
                            Dis2Frs.pop(index)
                        else:
                            print("finish sss")

                    else:#free_cluster_center和Dis2Frs没有点可选了
                        print("already finish")
                        self.finish_exploration_step[e] = True
                        self.finish_exploration[e] = True
                        map_goal[e] = self.agent_pos[e]

                        # for m in range(0,3):
                        #     self.finish_exploration_step[m] = True
                        #     self.finish_exploration[m] = True
                        #     map_goal[m] = self.agent_pos[m]
                            
                        # global_plan_accumulate[e] = self.agent_pos[e]
                        break
            elif self.finish_exploration[e] == True:
                map_goal[e] = self.agent_pos[e]
                self.finish_exploration_step[e] = True


        if self.finish_exploration != [True,True,True] and self.stop == False:
            return np.array(map_goal),global_plan_accumulate
        else:
            return np.array([]),[]

    def step_for_cost(self):
        obs = []
        flag = False
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros((self.num_agents, self.width, self.height))
        delta_reward_each_map = np.zeros((self.num_agents, self.width, self.height))
        reward_explored_each_map = np.zeros((self.num_agents, self.width, self.height))
        explored_all_map = np.zeros((self.width, self.height))
        obstacle_all_map = np.zeros((self.width, self.height))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(np.zeros((self.width, self.height)))
            self.obstacle_each_map_t.append(np.zeros((self.width, self.height)))
        self.finish_exploration_step = [False,False,False]
        action,global_plan_accumulate = self.get_goal_for_cost()
        if self.stop == True:
            self.finish_exploration = [True, True, True]
        for i in range(self.num_agents):
            if self.finish_exploration == [True,True,True] or self.stop == True:
                print("finish exploration")#无法再继续搜索下去了

                #后面的步骤是为了输出路径目标点的表格
                multiplied_list = []
                for sub_list in self.path_log:
                    # 创建一个空列表来存储当前子列表乘以10后的结果
                    multiplied_sub_list = []
                    sub_list = sub_list[::3]  #这一步主要是为了减少因为距离过短而出现问题的情况
                    # 内部循环遍历当前子列表中的每个二维元组
                    for tuple_item in sub_list:
                        # 将当前二维元组中的每个数乘以10，并添加到乘以10后的二维元组中
                        multiplied_tuple = tuple(x / self.scale_factor for x in tuple_item)
                        # 将乘以10后的二维元组添加到当前子列表的结果列表中
                        multiplied_sub_list.append(multiplied_tuple)
                    # 将当前子列表乘以10后的结果添加到最终结果列表中
                    multiplied_list.append(multiplied_sub_list)

                robot_0_path = [item for sublist in multiplied_list[0] for item in sublist]
                robot_1_path = [item for sublist in multiplied_list[1] for item in sublist]
                robot_2_path = [item for sublist in multiplied_list[2] for item in sublist]
                print("robot_0_path:",robot_0_path)
                print("robot_1_path:",robot_1_path)
                print("robot_2_path:",robot_2_path)
                # df.to_excel('path.xlsx')
                break

            elif self.stop == False:
                global_plan = global_plan_accumulate[i]
                pose = self.naive_local_planner(global_plan)
                # pose = pose[1:]
                random_integer = random.randint(1, 8)
                if i == 0: integer = 2
                if i == 1: integer = 5
                if i == 2: integer = 8
                if len(pose)>=8:
                    pose = pose[1:random_integer] #这一步主要是为了让它们的路径分开
                else:
                    pose = pose[1:]
                if pose == []:
                    self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
                    self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])

                    reward_explored_each_map[i] = self.explored_each_map[i].copy()
                    reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1

                    reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
                    reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1


                    delta_reward_each_map[i] = reward_explored_each_map[i]

                    each_agent_rewards.append(
                        (np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
                    self.previous_explored_each_map[i] = self.explored_each_map[i]

                    # for i in range(self.num_agents):
                    explored_all_map = np.maximum(self.explored_each_map[i], explored_all_map)
                    obstacle_all_map = np.maximum(self.obstacle_each_map[i], obstacle_all_map)

                else:#pose不是空集
                    for goal_pos in pose:#取出来每一步的路径
                        if goal_pos[0][0] - self.agent_pos[i][0] == 1:
                            self.agent_dir[i] = 1   #下
                        if goal_pos[0][0] - self.agent_pos[i][0] == -1:
                            self.agent_dir[i] = 0   #上
                        if goal_pos[0][1] - self.agent_pos[i][1] == 1:
                            self.agent_dir[i] = 3   #右
                        if goal_pos[0][1] - self.agent_pos[i][1] == -1:
                            self.agent_dir[i] = 2   #左


                        self.agent_pos[i] = goal_pos[0]

                        self.path_log[i].append(self.agent_pos[i])

                        # map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map_mine(self.agent_pos[i],self.agent_dir[i],self.gt_map,self.sensor_range)
                        map_this_frame, (x_min, x_max), (y_min, y_max) = self.optimized_build_map_with_direction(self.agent_pos[i], self.agent_dir[i],
                                                                                       self.gt_map, self.sensor_range,i)
                        self.built_map[i] = self.merge_two_map(self.built_map[i], map_this_frame, [x_min, x_max],
                                                               [y_min, y_max])

                        obs.append(self.built_map[i])
                        current_agent_pos.append(self.agent_pos[i])
                        self.explored_each_map_t[i] = (self.built_map[i] != 205).astype(int)
                        self.obstacle_each_map_t[i] = (self.built_map[i] == 0).astype(int)

                    # for i in range(self.num_agents):
                        self.explored_each_map[i] = np.maximum(self.explored_each_map[i], self.explored_each_map_t[i])
                        self.obstacle_each_map[i] = np.maximum(self.obstacle_each_map[i], self.obstacle_each_map_t[i])

                        reward_explored_each_map[i] = self.explored_each_map[i].copy()
                        reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1

                        reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
                        reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1


                        delta_reward_each_map[i] = reward_explored_each_map[i]

                        each_agent_rewards.append(
                            (np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
                        self.previous_explored_each_map[i] = self.explored_each_map[i]

                    # for i in range(self.num_agents):
                        explored_all_map = np.maximum(self.explored_each_map[i], explored_all_map)
                        obstacle_all_map = np.maximum(self.obstacle_each_map[i], obstacle_all_map)

        temp = explored_all_map + obstacle_all_map
        self.complete_map = np.zeros(temp.shape)
        self.complete_map[temp == 2] = 0
        self.complete_map[temp == 1] = 254
        self.complete_map[temp == 0] = 205


        obs = np.array(obs)
        if self.finish_exploration != [True,True,True]:
           self.plot_map_with_path()

        return obs


def obstacle_inflation(map, radius, resolution):
    inflation_grid = math.ceil(radius / resolution)
    import copy
    inflation_map = copy.deepcopy(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 0:
                neighbor_list = get_neighbor(i, j, inflation_grid, map.shape[0], map.shape[1])
                for inflation_point in neighbor_list:
                    inflation_map[inflation_point[0], inflation_point[1]] = 0
    return inflation_map


def get_neighbor(x, y, radius, x_max, y_max):
    neighbor_list = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if x + i > -1 and x + i < x_max and y + j > -1 and y + j < y_max:
                neighbor_list.append([x + i, y + j])
    return neighbor_list

def on_space():
    print("Space key pressed!")

def use_cost_method_to_explore(agent_num, map_name):
    env = GridEnv(1,  agent_num, 1000, map_name,
                  scale_factor=0.1, sensor_range = 4, robot_size=0.4,
                  #不太清楚 如果原地图1个像素点代表1cm的话，这个单位就是m
                  agent_0_pos = [240, 360],
                  agent_1_pos = [200, 1820],
                  agent_2_pos = [520, 370],
                  visualization=True)
    env.reset_for_traditional()
    keyboard.add_hotkey('space', env.on_space)
    while (env.finish_exploration != [True,True,True]):
        env.step_for_cost()

# 注册一个监听器，当键盘上任何按键被按下或释放时触发回调函数

if __name__ == "__main__":
    method_name = sys.argv[1]
    agent_num = int(sys.argv[2])
    map_name = sys.argv[3]
    if method_name == 'cost':
        use_cost_method_to_explore(agent_num, map_name)


