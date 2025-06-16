import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import os
import glob
import random
import argparse
import pickle
import matplotlib.pyplot as plt
from prm import prm_planning

def stepSimulation(iter):
    for k in range(iter):
        p.stepSimulation()
    return

class Watchdog():
    def __init__(self,limit=200):
        self.count = 0
        self.limit=limit
        return
    
    def error(self):
        self.count+=1
        if self.count>self.limit:
            return True
        return False
    
    def reset(self):
        self.count=0
        return

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class NavigationSim(object):
    def __init__(self, offset):
        self.offset = np.array(offset)
        self.LINK_EE_OFFSET = 0.05
        self.initial_offset = 0.05
        self.workspace_limits = np.asarray([[-3.0, 10.0], [-9, 9], [-10, 10]])
        self._numGoals = np.random.randint(1, 6)
        self._numObstacles = np.random.randint(1, 6)
        self._blockRandom = 0.3

        self.future_trajectory = None
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.init_orn = [0.0, 0.0, 0, 1.0]
        self.key_commands = [0, 0, 0]

        x_line_id = p.addUserDebugLine([0, 0.5, 0.01], [1, 0.5, 0.01], [1, 0, 0])
        y_line_id = p.addUserDebugLine([1, 0.5, 0.01], [1, -0.5, 0.01], [1, 0, 0])

        self.avatar_path = os.path.join(pd.getDataPath(), "sphere_small.urdf")
        self.avatar = p.loadURDF(
            self.avatar_path,
            np.array([0, 0, 0]) + self.offset,
            self.init_orn,
            # useFixedBase=True,
            flags=self.flags,
            globalScaling=20,
        )

        self.plane_path = os.path.join(pd.getDataPath(), "plane.urdf")
        self.plane = p.loadURDF(
            self.plane_path,
            np.array([0, 0, -0.6]) + self.offset,
        )


        self.goal_ids, self.obstacle_ids= self.setting_objects(
            globalScaling=30
        )
        self.goal_ids = set(self.goal_ids)
        self.obstacle_ids = set(self.obstacle_ids)
        # self.escape_ids = set(self.escape_ids)
        self.control_dt = 0.01
        self.place_poses = [
            -0.00018899307178799063,
            -0.3069845139980316,
            0.48534566164016724,
        ]
        self.z_T = 0.1
        self.pid = PIDController(kp=2, ki=0.01, kd=0.2, dt=0.05)
        # self.reset()

        # self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        # self.broadcaster = tf2_ros.TransformBroadcaster()

        # frame_start_postition, frame_posture = p.getLinkState(self.panda,11)[4:6]
        # R_Mat = np.array(p.getMatrixFromQuaternion(frame_posture)).reshape(3,3)
        # x_axis = R_Mat[:,0]
        # x_end_p = (np.array(frame_start_postition) + np.array(x_axis*5)).tolist()
        # x_line_id = p.addUserDebugLine(frame_start_postition,x_end_p,[1,0,0])# y 轴
        # y_axis = R_Mat[:,1]
        # y_end_p = (np.array(frame_start_postition) + np.array(y_axis*5)).tolist()
        # y_line_id = p.addUserDebugLine(frame_start_postition,y_end_p,[0,1,0])# z轴
        # z_axis = R_Mat[:,2]
        # z_end_p = (np.array(frame_start_postition) + np.array(z_axis*5)).tolist()
        # z_line_id = p.addUserDebugLine(frame_start_postition,z_end_p,[0,0,1])

        return

    # def setting_objects(self,globalScaling):
    #     goal_ids = []
    #     obstacle_ids = []
    #     # blue cube
    #     object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
    #     # pos = np.array([[6,-7,0], [8.5,-3.5,0], [9.2,0,0], [8.5,3.5,0], [6,7, 0]])
    #     for i in range(self._numObjects):
    #         # if file in ['cube_5.sdf', 'cube_6.sdf']:
    #         #     continue
    #         rand_pos = np.random.rand(2)
    #         rand_pos[0] = rand_pos[0]*(self.workspace_limits[0,1] - self.workspace_limits[0,0]) + self.workspace_limits[0,0]
    #         rand_pos[1] = rand_pos[1]*(self.workspace_limits[1,1] - self.workspace_limits[1,0]) + self.workspace_limits[1,0]

    #         # rand_pos[1] = (2*rand_pos[0] - 1)*0.5
    #         rand_pos = np.append(rand_pos, 0.0)
    #         uid = p.loadURDF(object_path, rand_pos, self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=globalScaling)
    #         goal_ids.append(uid)
    #     return goal_ids, obstacle_ids

    def setting_objects(self, globalScaling):
        goal_ids = []
        obstacle_ids = []
        banned_radius = 3
        obstacle_extension = 3
        all_objects_pos = np.zeros((0, 2))
        # blue cube
        # object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        object_path = os.path.join(curr_dir, "obj", "goal_small.urdf")
        # object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
        # pos = np.array([[6,-7,0], [8.5,-3.5,0], [9.2,0,0], [8.5,3.5,0], [6,7, 0]])
        for i in range(np.random.randint(1,self._numGoals+1)):
            # if file in ['cube_5.sdf', 'cube_6.sdf']:
            #     continue
            while True:
                rand_pos = np.random.rand(2)
                rand_pos[0] = (
                    rand_pos[0]
                    * (self.workspace_limits[0, 1] - self.workspace_limits[0, 0])
                    + self.workspace_limits[0, 0]
                )
                rand_pos[1] = (
                    rand_pos[1]
                    * (self.workspace_limits[1, 1] - self.workspace_limits[1, 0])
                    + self.workspace_limits[1, 0]
                )

                if (
                    all_objects_pos.shape[0] != 0
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1
                ):
                    continue

                if np.linalg.norm(rand_pos) > banned_radius+obstacle_extension:
                    break
            all_objects_pos = np.concatenate(
                (all_objects_pos, rand_pos.reshape(-1, 2)), axis=0
            )

            # rand_pos[1] = (2*rand_pos[0] - 1)*0.5
            rand_pos = np.append(rand_pos, 0.0)
            uid = p.loadURDF(
                object_path,
                rand_pos,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=globalScaling,
            )
            goal_ids.append(uid)

        object_path = os.path.join(curr_dir, "obj", "obstacle_small.urdf")
        for i in range(np.random.randint(1,self._numObstacles+1)):
            # if file in ['cube_5.sdf', 'cube_6.sdf']:
            #     continue
            while True:
                rand_pos = np.random.rand(2)
                rand_pos[0] = (
                    rand_pos[0]
                    * (self.workspace_limits[0, 1] - self.workspace_limits[0, 0])
                    + self.workspace_limits[0, 0]
                )
                rand_pos[1] = (
                    rand_pos[1]
                    * (self.workspace_limits[1, 1] - self.workspace_limits[1, 0])
                    + self.workspace_limits[1, 0]
                )

                if (
                    all_objects_pos.shape[0] != 0
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1
                ):
                    continue

                if np.linalg.norm(rand_pos) > banned_radius and np.linalg.norm(rand_pos) < banned_radius + obstacle_extension:
                    break
            all_objects_pos = np.concatenate(
                (all_objects_pos, rand_pos.reshape(-1, 2)), axis=0
            )

            # rand_pos[1] = (2*rand_pos[0] - 1)*0.5
            rand_pos = np.append(rand_pos, 0.0)
            uid = p.loadURDF(
                object_path,
                rand_pos,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=globalScaling,
            )
            obstacle_ids.append(uid)
        return goal_ids, obstacle_ids

    def reset(self):
        # remove previous objects from sim and list
        for id in self.goal_ids:
            self.remove(id)
        for id in self.obstacle_ids:
            self.remove(id)

        avatar.remove(self.avatar)

        self.goal_ids = set()
        self.obstacle_ids = set()

        self.avatar = p.loadURDF(
            self.avatar_path,
            self.offset,
            self.init_orn,
            # useFixedBase=True,
            flags=self.flags,
            globalScaling=20,
        )

        # When reaseting, change num of obhjects as well

        self._numGoals = np.random.randint(1, 6)
        self._numObstacles = np.random.randint(2, 8)

        if len(self.goal_ids) == 0:
            self.goal_ids, self.obstacle_ids = self.setting_objects(
                globalScaling=30
            )
            self.goal_ids = set(self.goal_ids)
            self.obstacle_ids = set(self.obstacle_ids)

    def remove(self, tar_id):
        try:
            p.removeBody(tar_id)
        except:
            print("No object with id: ", tar_id)
        return

    def calculate_velocity(self, tar_id):
        avatar_pos, _ = p.getBasePositionAndOrientation(self.avatar)
        object_pos, _ = p.getBasePositionAndOrientation(tar_id)

        velo = np.array(object_pos) - np.array(avatar_pos)
        velo += (2 * np.random.rand(velo.shape[0]) - 1) * (velo)
        velo = velo.clip(-3, 3)
        velo[-1] = 0.0
        return velo

    def move(self, tar_id):
        success = False
        velo = self.calculate_velocity(tar_id)
        object_pos, _ = p.getBasePositionAndOrientation(tar_id)

        p.resetBaseVelocity(avatar.avatar, linearVelocity=velo)
        avatar_pos, _ = p.getBasePositionAndOrientation(self.avatar)
        if np.abs(np.array(avatar_pos) - np.array(object_pos)).mean() < 1:
            success = True
        return success

    def arrived(self, tar_id):
        success = False
        object_pos, _ = p.getBasePositionAndOrientation(tar_id)
        avatar_pos, _ = p.getBasePositionAndOrientation(self.avatar)
        if np.abs(np.array(avatar_pos) - np.array(object_pos)).mean() < 1:
            success = True
        return success

    def move2pose(self, object_pos, threshold=1, noise=True):
        success = False
        avatar_pos, _ = p.getBasePositionAndOrientation(self.avatar)
        velo = np.array(object_pos) - np.array(avatar_pos)
        velo = self.pid.compute(velo)
        if np.random.random() < 0.5 and noise:
            velo += (2 * np.random.rand(velo.shape[0]) - 1) * (velo + 1)
        velo = velo.clip(-3, 3)
        velo[-1] = 0.0

        p.resetBaseVelocity(avatar.avatar, linearVelocity=velo)
        avatar_pos, _ = p.getBasePositionAndOrientation(self.avatar)
        if np.abs(np.array(avatar_pos) - np.array(object_pos)).mean() < threshold:
            success = True
        return success

    def random_select_object(self):
        tar_id = random.choice(list(self.goal_ids))
        return tar_id

    def get_obstacle_poses(self):
        poses = []
        for id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(id)
            poses.append(pos)
        return poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMI Navigation Data Collection")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode")
    parser.add_argument("--total_data_num", type=int, default=1000000, help="Total number of trajectories to collect")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval to save data")
    parser.add_argument("--data_path", type=str, default="data.pkl", help="Path to save trajectory data")
    parser.add_argument("--noise", action="store_false", help="Adding noise in velocity")
    args = parser.parse_args()
    # rospy.init_node("bmi_sim")
    if args.gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)

    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=-90,
        cameraPitch=-20,
        cameraTargetPosition=[-0.0, -0.0, 1.5],
    )
    timeStep = 1.0 / 100.0
    steps = 5
    p.setTimeStep(timeStep)
    p.setRealTimeSimulation(1)

    index = 1
    total_data_num = args.total_data_num
    save_interval = args.save_interval
    # traj_log = Queue()

    # rospy.Rate(2)

    avatar = NavigationSim([0, 0, 0])

    data_path = "data_bmi2d_w_goals_colavoid.json"
    if os.path.exists(data_path):
        with open(data_path, "rb") as fp:
            data_dict = pickle.load(fp)
        # data_dict["frequency"] = int(1 / timeStep) // 5
        data = data_dict["data"]
        goal_list = data_dict["goals"]
        obstacle_list = data_dict["obstacles"]
    else:
        data = []
        goal_list = []
        obstacle_list = []
    wall_x = np.linspace(0, 9, 2)
    wall_y = np.linspace(-9, 9, 2)

    wall_ox, wall_oy = np.meshgrid(wall_x, wall_y)
    wall_ox = wall_ox.reshape(-1).tolist()
    wall_oy = wall_oy.reshape(-1).tolist()

    watchdog = Watchdog(140)

    while True:
        trajectory = []
        goals_per_traj = []
        obstacles_per_traj = []
        try:
            tar_id = avatar.random_select_object()
        except:
            avatar.reset()
        resolution_scale = 2
        # simplegrid = np.zeros((12*resolution_scale, 20*resolution_scale))
        ox = []
        oy = []
        for goal_id in avatar.goal_ids:
            object_pos, _ = p.getBasePositionAndOrientation(goal_id)
            object_pos = list(object_pos)
            goals_per_traj.append(object_pos)
            # goals_per_traj += object_pos
            if tar_id != goal_id:
                # ox.append(object_pos[0])
                # oy.append(object_pos[1])
                continue
                # simplegrid[simplegrid.shape[0]-int(object_pos[0]*resolution_scale), simplegrid.shape[1]-int(resolution_scale*(object_pos[1]+10) )] = 1
            else:
                real_target = object_pos
        obstacles = np.array(avatar.get_obstacle_poses())
        obstacles_per_traj = obstacles.tolist()
        ox += obstacles[:, 0].tolist()
        oy += obstacles[:, 1].tolist()
        rx, ry = prm_planning(
            0.0,
            0.0,
            real_target[0],
            real_target[1],
            ox + wall_ox,
            oy + wall_oy,
            1.5,
            rng=None,
        )

        # fig = plt.figure()
        # plt.plot(rx, ry)
        # plt.show()

        change_of_intent = np.random.random() < 0.8

        try:
            assert rx, "Cannot find path"
        except:
            # while True:
            #     if len(avatar.goal_ids) == 0:
            #         break
            #     id = avatar.random_select_object()
            #     avatar.remove(id)
            avatar.reset()
            continue

        success = False
        while True:
            for p_idx in range(len(rx)):
                success = False
                threshold = 0.2 if p_idx == len(rx) - 1 else 0.5
                while True:
                    success = avatar.move2pose(
                        (rx[-p_idx - 1], ry[-p_idx - 1], 0.0), threshold, args.noise
                    )
                    avatar_pos, _ = p.getBasePositionAndOrientation(avatar.avatar)
                    trajectory.append(list(avatar_pos))
                    stepSimulation(steps)
                    time.sleep(0.05)

                    # if change_of_intent and np.random.random() < 0.1:
                    #     tar_id = avatar.random_select_object()
                    #     print("change")
                    #     change_of_intent = False
                    if watchdog.error():
                        break
                    if success == True:
                        break

            if watchdog.error():
                break
            if avatar.arrived(tar_id):
                # avatar.remove(tar_id)
                break
            # else:
            #     print()
        if watchdog.error():
            avatar.reset()
            watchdog.reset()
            continue

        # index += 1
        data.append(trajectory)
        goal_list.append(goals_per_traj)
        obstacle_list.append(obstacles_per_traj)
        avatar.reset()

        if len(data) > total_data_num or len(data)%save_interval==0:
            print("sample:", len(data))
            data_dict = {}
            data_dict["frequency"] = int(1 / timeStep) // 5
            data_dict["data"] = data
            data_dict["goals"] = goal_list
            data_dict["obstacles"] = obstacle_list
            with open(data_path, "wb") as fp:
                pickle.dump(data_dict, fp)
            if len(data) >= total_data_num:
                break

    print()