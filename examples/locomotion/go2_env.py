import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def compute_agent_centers(field_size: float, num_agents: int):
    """
    正方形のfieldをエージェント数分だけ等しく正方形に分割し、
    各エージェントの領域の中心座標を計算する。
    
    :param field_size: フィールドの一辺の長さ
    :param num_agents: エージェントの数（完全な平方数である必要がある）
    :return: 各エージェント領域の中心座標のリスト
    """
    # num_agents が完全な平方数であることを確認
    grid_size = int(np.sqrt(num_agents))
    if grid_size ** 2 != num_agents:
        raise ValueError("num_agents must be a perfect square (e.g., 4, 9, 16, ...).")
    
    # 各小領域の一辺の長さ
    cell_size = field_size / grid_size
    
    # 中心座標を計算（フィールドの中心を (0,0) にする）
    centers = []
    offset = 0
    # offset = field_size / 2
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (j + 0.5) * cell_size - offset
            center_y = (i + 0.5) * cell_size - offset
            centers.append((center_x, center_y, 0.42))
    
    return centers


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, camera_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda", debug=False):
        self.device = torch.device(device)
        self.debug = debug

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.img_obs_dim = obs_cfg["img_obs_dim"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.camera_cfg = camera_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        print(self.camera_cfg)
        self.fixed_camera_render_cfg = {"depth": camera_cfg["fixed_camera"]["use_depth"]}
        del self.camera_cfg["fixed_camera"]["use_depth"]
        if "follower_camera" in camera_cfg:
            self.follower_camera_render_cfg = {"depth": camera_cfg["follower_camera"]["use_depth"]}
            del self.camera_cfg["follower_camera"]["use_depth"]
        if "head_camera" in camera_cfg:
            self.head_camera_render = {"rgb": camera_cfg["head_camera"]["use_rgb"], "depth": camera_cfg["head_camera"]["use_depth"]}
            del self.camera_cfg["head_camera"]["use_rgb"]
            del self.camera_cfg["head_camera"]["use_depth"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=self.env_cfg["n_rendered_envs"]),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        # defalut plane
        if self.debug:
            self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", pos=(0,0,0), fixed=True)) # OK1

        # self.scene.add_entity(gs.morphs.Mesh(file="stair/STAIRS.stl", pos=(2,0,0), euler=(0,0,-90), fixed=True, scale=0.2))
        # self.scene.add_entity(gs.morphs.Mesh(file="terrain-generator/results/generated_terrain/mesh_0/mesh.obj", pos=(20,-0.2,0.8), fixed=True, scale=1.0, convexify=False)) # NG1
        # self.scene.add_entity(gs.morphs.Mesh(file="terrain-generator/results/generated_terrain/mesh_0/mesh.obj", pos=(23,-0.2,0.5), fixed=True, scale=1.0))
        # self.scene.add_entity(gs.morphs.Mesh(file="terrain-generator/results/generated_terrain/mesh_0/mesh.obj", pos=(23,-0.2,-10), fixed=True, scale=1.0)) # OK1
        # self.scene.add_entity(gs.morphs.Mesh(file="terrain-generator/results/generated_terrain/mesh_0/mesh.obj", pos=(0,0,0), fixed=True, scale=10.0))
        # self.scene.add_entity(gs.morphs.Mesh(file="terrain-generator/results/generated_terrain/mesh_1/mesh.obj", pos=(0,0.5,0.5), fixed=True, scale=1.0, convexify=False)) # NG2
        # self.scene.add_entity(gs.morphs.Terrain(file="terrain-generator/results/generated_terrain/mesh_0/mesh.obj", pos=(0,0,-0.1)))
        # horizontal_scale = 0.25
        # vertical_scale = 0.005
        # height_field = np.zeros([40, 40])
        # heights_range = np.arange(-10, 20, 10)
        # height_field[5:35, 5:35] = np.random.choice(heights_range, (30, 30))
        # ########################## entities ##########################
        # terrain = self.scene.add_entity(
        #     morph=gs.morphs.Terrain(
        #         horizontal_scale=horizontal_scale,
        #         vertical_scale=vertical_scale,
        #         height_field=height_field,
        #         pos=(-5, -5, 0.),
        #     ),
        # )
        # horizontal_scale = 0.25
        # vertical_scale = 0.005
        # self.scene.add_entity(
        #     morph=gs.morphs.Terrain(
        #         horizontal_scale=horizontal_scale,
        #         vertical_scale=vertical_scale,
        #         pos=(-15, -15, 0.),
        #     ),
        # )
        # self.scene.add_entity(
        #     morph=gs.morphs.Terrain(
        #         n_subterrains=(2, 2),
        #         horizontal_scale=horizontal_scale,
        #         vertical_scale=vertical_scale,
        #         subterrain_types=[
        #             ["flat_terrain", "random_uniform_terrain"],
        #             ["stepping_stones_terrain", "holey_terrain"],
        #         ],
        #         pos=(0, 0, 0.),
        #     ),
        # )
        
        # final terrain
        if not self.debug:
            horizontal_scale = 0.25
            vertical_scale = 0.005
            num_fields = np.sqrt(num_envs)
            assert num_fields.is_integer(), "num_envs must be a perfect square (e.g., 4, 9, 16, ...)."
            num_fields = int(num_fields)
            subterrain_types = [["holey_terrain"]*num_fields]*num_fields
            self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    n_subterrains=(num_fields, num_fields),
                    horizontal_scale=horizontal_scale,
                    vertical_scale=vertical_scale,
                    subterrain_types=subterrain_types,
                    pos=(0, 0, 0.),
                ),
            )

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # calc center of agents 
        if not self.debug:
            field_size = 12 * num_fields 
            num_agents = num_envs
            self.base_init_poses = torch.tensor(compute_agent_centers(field_size, num_agents), device=self.device)
            print(self.base_init_poses)

        # add fixed camera
        if self.debug:
            self.fixed_camera = self.scene.add_camera(**self.camera_cfg["fixed_camera"],)
        else:
            self.camera_cfg["fixed_camera"]["pos"] = (field_size/2, field_size/2, 20)
            self.fixed_camera = self.scene.add_camera(**self.camera_cfg["fixed_camera"],)

        # add follower camera
        if "follower_camera" in self.camera_cfg:
            self.follower_camera = self.scene.add_camera(**self.camera_cfg["follower_camera"],)
            # follow the robot at a fixed height and orientation 
            self.follower_camera.follow_entity(self.robot, fixed_axis=(None, None, None), smoothing=0.5, fix_orientation=False)

        # add head cameras
        if "head_camera" in self.camera_cfg:
            self.head_cameras = []
            for i in range(num_envs):
                self.head_cameras.append(self.scene.add_camera(**self.camera_cfg["head_camera"]))
            theta_x = np.deg2rad(90)
            theta_y = np.deg2rad(-90)
            # theta_z = np.deg2rad(90)
            Rx = np.array([
                [ 1, 0, 0, 0],
                [ 0, np.cos(theta_x), -np.sin(theta_x), 0],
                [ 0, np.sin(theta_x), np.cos(theta_y), 0],
                [               0, 0,               0, 1]
            ])
            Ry = np.array([
                [ np.cos(theta_y), 0, np.sin(theta_y), 0],
                [               0, 1,               0, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                [               0, 0,               0, 1]
            ])
            # Rz = np.array([
            #     [ np.cos(theta_z), -np.sin(theta_z), 0, 0],
            #     [ np.sin(theta_z),  np.cos(theta_z), 0, 0],
            #     [               0,                0, 1, 0],
            #     [               0,                0, 0, 1]
            # ])
            # 「Z軸回転 → Y軸回転」をまとめた回転行列
            R = Rx @ Ry
            offset_T = np.eye(4)
            offset_T[:, 3] = [0, 0.1, -0.26, 1] # オフセット行列の平行移動成分を設定 y, z, -x world座標系(右手系)
            # offset_T[0, 3] = 0.1
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            offset_T = R @ offset_T  # オフセット行列にZ回転を適用
            for i in range(num_envs):
                self.head_cameras[i].attach(self.robot, offset_T, i)


        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.img_obs_buf = torch.zeros((self.num_envs, *self.img_obs_dim), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        if hasattr(self, "fixed_camera"):
            # fixed_rgb, fixed_depth, fixed_seg, fixed_normal = self.fixed_camera.render(depth=True, segmentation=True, normal=True, colorize_seg=True)
            self.fixed_camera.render(**self.fixed_camera_render_cfg)

        if hasattr(self, "follower_camera"):
            # follow_rgb, follow_depth, follow_seg, follow_normal = self.follower_camera.render(depth=True, segmentation=True, normal=True, colorize_seg=True)
            self.follower_camera.render(**self.follower_camera_render_cfg) 

        if hasattr(self, "head_cameras"):
            head_rgbs = []
            head_depths = []
            # head_segs = []
            # head_normals = []
            for head_camera in self.head_cameras:
                head_rgb, head_depth, head_seg, head_normal = head_camera.render(**self.head_camera_render)
                head_rgbs.append(head_rgb)
                head_depths.append(head_depth)
                # head_segs.append(head_seg)
                # head_normals.append(head_normal)
            head_rgbs = np.array(head_rgbs)
            head_depths = np.array(head_depths)
            # head_segs = np.array(head_segs)
            # head_normals = np.array(head_normals)

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        if hasattr(self, "head_cameras"):
            import cv2
            def resize_batch(batch, new_size):
                return np.array([cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR) for img in batch])
            # リサイズを適用する関数
            def resize_depth_batch(depth_batch, new_size):
                resized_batch = np.zeros((depth_batch.shape[0], new_size[0], new_size[1]), dtype=np.float32)
                for i in range(depth_batch.shape[0]):
                    resized_batch[i] = cv2.resize(depth_batch[i], (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
                return resized_batch
            # resized_head_rgbs = resize_batch(head_rgbs, (128, 128))
            # resized_head_depths = resize_depth_batch(head_depths, (224, 224))
            # self.img_obs_buf = torch.cat(
            #     [
            #         torch.from_numpy(resized_head_rgbs.copy()).permute(0, 3, 1, 2).float() / 255.0,
            #         torch.from_numpy(resized_head_depths.copy()).unsqueeze(1).float() / 255.0,
            #     ],
            #     axis=1,
            # ) 
            if self.head_camera_render["rgb"] and self.head_camera_render["depth"]:
                self.img_obs_buf = torch.cat(
                    [
                        torch.from_numpy(head_rgbs.copy()).permute(0, 3, 1, 2).float() / 255.0,
                        torch.from_numpy(head_depths.copy()).unsqueeze(1).float() / 255.0,
                    ],
                    axis=1,
                ) 
            elif self.head_camera_render["depth"]:
                # self.img_obs_buf = torch.from_numpy(resized_head_depths.copy()).unsqueeze(1).float() / 255.0
                self.img_obs_buf = torch.from_numpy(head_depths.copy()).unsqueeze(1).float() / 255.0

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras, self.img_obs_buf

    def get_observations(self):
        return self.obs_buf, self.img_obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        if self.debug:
            self.base_pos[envs_idx] = self.base_init_pos
        else:
            self.base_pos[envs_idx] = self.base_init_poses[envs_idx]
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        print("reset")
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None, self.img_obs_buf

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
