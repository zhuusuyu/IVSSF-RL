import math
import os
import threading
import time

import gymnasium as gym
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SetModelState, SpawnModel
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
from rospy.exceptions import ROSTimeMovedBackwardsException
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty as EmptySrv

try:
    from .social_state import extract_teacher_state
except ImportError:
    from social_state import extract_teacher_state


class SocialTeacherRosEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    SCENE_CONFIGS = {
        "teacher_pretrain": {
            "ped_proxy_names": ("ped_teacher_proxy_1", "ped_teacher_proxy_2"),
            "start_points": [(-3.2, 0.0, 0.0), (-3.1, 0.9, 0.0), (-3.1, -0.9, 0.0)],
            "goal_points": [(3.0, 0.0), (2.9, 0.9), (2.9, -0.9)],
        },
        "teacher_cross": {
            "ped_proxy_names": ("ped_teacher_cross_proxy_1", "ped_teacher_cross_proxy_2"),
            "start_points": [(-3.0, 0.0, 0.0), (-3.0, 0.8, 0.0), (-3.0, -0.8, 0.0)],
            "goal_points": [(2.8, 0.0), (2.8, 0.8), (2.8, -0.8)],
        },
        "teacher_cross_dual_lane": {
            "ped_proxy_names": (
                "ped_teacher_cross_dual_proxy_h1",
                "ped_teacher_cross_dual_proxy_h2",
                "ped_teacher_cross_dual_proxy_v1",
                "ped_teacher_cross_dual_proxy_v2",
            ),
            "start_points": [
                (-3.0, 0.0, 0.0),
                (3.0, 0.0, math.pi),
                (0.0, -3.0, math.pi * 0.5),
                (0.0, 3.0, -math.pi * 0.5),
            ],
            "goal_points": [
                (-3.0, 0.0),
                (3.0, 0.0),
                (0.0, -3.0),
                (0.0, 3.0),
            ],
            "start_arm_ids": ("west", "east", "south", "north"),
            "goal_arm_ids": ("west", "east", "south", "north"),
        },
        "hybrid": {
            "ped_proxy_names": ("ped_proxy_1", "ped_proxy_2", "ped_proxy_3", "ped_proxy_4"),
            "start_points": [(-3.5, 0.0, 0.0), (-3.5, 1.6, 0.0), (-3.5, -1.6, 0.0)],
            "goal_points": [(3.5, 0.0), (3.5, 1.6), (3.5, -1.6)],
        },
        "hybrid_8ped": {
            "ped_proxy_names": (
                "ped_proxy_1",
                "ped_proxy_2",
                "ped_proxy_3",
                "ped_proxy_4",
                "ped_proxy_5",
                "ped_proxy_6",
                "ped_proxy_7",
                "ped_proxy_8",
            ),
            "start_points": [(-3.5, 0.0, 0.0), (-3.5, 1.6, 0.0), (-3.5, -1.6, 0.0)],
            "goal_points": [(3.5, 0.0), (3.5, 1.6), (3.5, -1.6)],
        },
        "wall_8ped_opposite": {
            "ped_proxy_names": (
                "ped_wall8_proxy_1",
                "ped_wall8_proxy_2",
                "ped_wall8_proxy_3",
                "ped_wall8_proxy_4",
                "ped_wall8_proxy_5",
                "ped_wall8_proxy_6",
                "ped_wall8_proxy_7",
                "ped_wall8_proxy_8",
            ),
            "start_points": [
                (-3.5, -1.6, 0.0),
                (-3.5, 0.0, 0.0),
                (-3.5, 1.6, 0.0),
                (3.5, -1.6, math.pi),
                (3.5, 0.0, math.pi),
                (3.5, 1.6, math.pi),
                (-1.6, -3.5, math.pi * 0.5),
                (0.0, -3.5, math.pi * 0.5),
                (1.6, -3.5, math.pi * 0.5),
                (-1.6, 3.5, -math.pi * 0.5),
                (0.0, 3.5, -math.pi * 0.5),
                (1.6, 3.5, -math.pi * 0.5),
            ],
            "goal_points": [
                (-3.5, -1.6),
                (-3.5, 0.0),
                (-3.5, 1.6),
                (3.5, -1.6),
                (3.5, 0.0),
                (3.5, 1.6),
                (-1.6, -3.5),
                (0.0, -3.5),
                (1.6, -3.5),
                (-1.6, 3.5),
                (0.0, 3.5),
                (1.6, 3.5),
            ],
            "start_arm_ids": (
                "west",
                "west",
                "west",
                "east",
                "east",
                "east",
                "south",
                "south",
                "south",
                "north",
                "north",
                "north",
            ),
            "goal_arm_ids": (
                "west",
                "west",
                "west",
                "east",
                "east",
                "east",
                "south",
                "south",
                "south",
                "north",
                "north",
                "north",
            ),
            "goal_indices_by_start_arm": {
                "west": (3, 4, 5),
                "east": (0, 1, 2),
                "south": (9, 10, 11),
                "north": (6, 7, 8),
            },
        },
    }

    def __init__(
        self,
        render_mode=None,
        dt=0.05,
        action_duration=0.2,
        v_max=0.22,
        w_max=1.5,
        goal_radius=0.22,
        timeout_steps=180,
        scene_mode="hybrid",
        robot_model_name=None,
        ped_proxy_names=None,
        goal_model_name="teacher_goal",
        start_points=None,
        goal_points=None,
        progress_scale=40.0,
        arrive_reward=200.0,
        collision_penalty=-200.0,
        step_penalty=-0.5,
        collision_distance=0.16,
        d_safe=0.55,
        epsilon_1=80.0,
        teacher_angle_config=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = float(dt)
        self.action_duration = float(action_duration)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.goal_radius = float(goal_radius)
        self.timeout_steps = int(timeout_steps)
        self.scene_mode = str(scene_mode)
        self.goal_model_name = goal_model_name
        self.progress_scale = float(progress_scale)
        self.arrive_reward = float(arrive_reward)
        self.collision_penalty = float(collision_penalty)
        self.step_penalty = float(step_penalty)
        self.collision_distance = float(collision_distance)
        self.d_safe = float(d_safe)
        self.epsilon_1 = float(epsilon_1)
        self.teacher_angle_config = teacher_angle_config

        if self.scene_mode not in self.SCENE_CONFIGS:
            raise ValueError(f"Unsupported scene_mode: {self.scene_mode}")
        scene_cfg = self.SCENE_CONFIGS[self.scene_mode]

        self.robot_model_name = robot_model_name or f"turtlebot3_{os.environ.get('TURTLEBOT3_MODEL', 'burger')}"
        self.ped_proxy_names = tuple(ped_proxy_names or scene_cfg["ped_proxy_names"])

        self.start_points = list(start_points or scene_cfg["start_points"])
        self.goal_points = list(goal_points or scene_cfg["goal_points"])
        self.start_arm_ids = list(scene_cfg.get("start_arm_ids", [None] * len(self.start_points)))
        self.goal_arm_ids = list(scene_cfg.get("goal_arm_ids", [None] * len(self.goal_points)))
        goal_indices_cfg = scene_cfg.get("goal_indices_by_start_arm")
        self.goal_indices_by_start_arm = (
            {
                str(arm): tuple(int(goal_idx) for goal_idx in goal_indices)
                for arm, goal_indices in goal_indices_cfg.items()
            }
            if goal_indices_cfg is not None
            else None
        )
        self.max_goal_distance = max(
            math.hypot(gx - sx, gy - sy)
            for sx, sy, _ in self.start_points
            for gx, gy in self.goal_points
        ) + 0.5
        if len(self.start_arm_ids) != len(self.start_points):
            raise ValueError("start_arm_ids must align with start_points")
        if len(self.goal_arm_ids) != len(self.goal_points):
            raise ValueError("goal_arm_ids must align with goal_points")
        if self.goal_indices_by_start_arm is not None:
            start_arms = {arm for arm in self.start_arm_ids if arm is not None}
            missing_arms = sorted(start_arms - set(self.goal_indices_by_start_arm.keys()))
            if missing_arms:
                raise ValueError(
                    "goal_indices_by_start_arm must define goal candidates for all non-null start arms: "
                    + ", ".join(missing_arms)
                )
            for arm, goal_indices in self.goal_indices_by_start_arm.items():
                if not goal_indices:
                    raise ValueError(f"goal_indices_by_start_arm['{arm}'] must not be empty")
                for goal_idx in goal_indices:
                    if goal_idx < 0 or goal_idx >= len(self.goal_points):
                        raise ValueError(
                            f"goal_indices_by_start_arm['{arm}'] contains out-of-range goal index {goal_idx}"
                        )

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -self.w_max], dtype=np.float32),
            high=np.array([self.v_max, self.w_max], dtype=np.float32),
            dtype=np.float32,
        )

        if not rospy.core.is_initialized():
            rospy.init_node("social_teacher_env", anonymous=True, disable_signals=True)

        self._lock = threading.Lock()
        self._last_scan = None
        self._last_odom = None
        self._model_states = {}
        self._have_scan = False
        self._have_odom = False

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=1)
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1)

        rospy.wait_for_service("/gazebo/reset_world")
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/delete_model")

        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", EmptySrv)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

        self.goal_xy = np.zeros(2, dtype=np.float32)
        self.prev_goal_distance = None
        self.prev_scan_ranges = None
        self.episode_min_human_distance = None
        self.steps = 0
        self.episode_seed = 0
        self._goal_sdf = self._make_goal_sdf()

    def _scan_cb(self, msg):
        with self._lock:
            self._last_scan = msg
            self._have_scan = True

    def _odom_cb(self, msg):
        with self._lock:
            self._last_odom = msg
            self._have_odom = True

    def _model_states_cb(self, msg):
        with self._lock:
            self._model_states = {name: pose for name, pose in zip(msg.name, msg.pose)}

    def _wait_for_messages(self, timeout=5.0):
        t0 = time.time()
        while not rospy.is_shutdown():
            with self._lock:
                if self._have_scan and self._have_odom:
                    return True
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        return False

    def _wait_for_scene_ready(self, timeout=10.0):
        t0 = time.time()
        required_models = {self.robot_model_name, *self.ped_proxy_names}
        while not rospy.is_shutdown():
            with self._lock:
                have_scan = self._have_scan
                have_odom = self._have_odom
                model_names = set(self._model_states.keys())
            missing_models = sorted(required_models - model_names)
            if have_scan and have_odom and not missing_models:
                return True, []
            if time.time() - t0 > timeout:
                missing = []
                if not have_scan:
                    missing.append("/scan")
                if not have_odom:
                    missing.append("/odom")
                missing.extend(missing_models)
                return False, missing
            time.sleep(0.05)
        return False, ["ros_shutdown"]

    @staticmethod
    def _wall_sleep(duration):
        end_time = time.time() + max(0.0, float(duration))
        while time.time() < end_time and not rospy.is_shutdown():
            try:
                time.sleep(min(0.02, max(0.0, end_time - time.time())))
            except ROSTimeMovedBackwardsException:
                continue

    def _make_goal_sdf(self):
        return f"""
<sdf version='1.6'>
  <model name='{self.goal_model_name}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry><cylinder><radius>0.18</radius><length>0.02</length></cylinder></geometry>
        <material><ambient>0 0.8 0 1</ambient><diffuse>0 0.8 0 1</diffuse></material>
        <pose>0 0 0.01 0 0 0</pose>
      </visual>
    </link>
  </model>
</sdf>
""".strip()

    def _stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def _delete_goal(self):
        try:
            self.delete_model(self.goal_model_name)
        except rospy.ServiceException:
            pass

    def _spawn_goal(self, x_val, y_val):
        pose = Pose()
        pose.position = Point(x=x_val, y=y_val, z=0.0)
        pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        return self.spawn_model(
            model_name=self.goal_model_name,
            model_xml=self._goal_sdf,
            robot_namespace="",
            initial_pose=pose,
            reference_frame="world",
        )

    def _set_named_model_pose(self, model_name, x_val, y_val, yaw_val, z_val=0.0):
        quat = Quaternion(0.0, 0.0, math.sin(yaw_val * 0.5), math.cos(yaw_val * 0.5))
        state = ModelState()
        state.model_name = model_name
        state.pose = Pose(position=Point(x=x_val, y=y_val, z=z_val), orientation=quat)
        state.twist = Twist()
        state.reference_frame = "world"
        return self.set_model_state(state)

    def _place_goal(self, x_val, y_val):
        self._delete_goal()
        self._wall_sleep(0.05)
        for _ in range(3):
            try:
                response = self._spawn_goal(x_val, y_val)
                if getattr(response, "success", True):
                    return True
            except rospy.ServiceException:
                pass
            self._wall_sleep(0.05)

        try:
            response = self._set_named_model_pose(self.goal_model_name, x_val, y_val, 0.0, z_val=0.0)
            return bool(getattr(response, "success", True))
        except rospy.ServiceException:
            return False

    def _set_robot_pose(self, x_val, y_val, yaw_val):
        quat = Quaternion(0.0, 0.0, math.sin(yaw_val * 0.5), math.cos(yaw_val * 0.5))
        state = ModelState()
        state.model_name = self.robot_model_name
        state.pose = Pose(position=Point(x=x_val, y=y_val, z=0.0), orientation=quat)
        state.twist = Twist()
        state.reference_frame = "world"
        return self.set_model_state(state)

    def _current_messages(self):
        with self._lock:
            return self._last_scan, self._last_odom, dict(self._model_states)

    @staticmethod
    def _yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _robot_xy_yaw(self, odom_msg):
        pose = odom_msg.pose.pose
        return (
            float(pose.position.x),
            float(pose.position.y),
            float(self._yaw_from_quaternion(pose.orientation)),
        )

    def _goal_distance(self, odom_msg):
        robot_x, robot_y, _ = self._robot_xy_yaw(odom_msg)
        return float(math.hypot(self.goal_xy[0] - robot_x, self.goal_xy[1] - robot_y))

    def _min_human_distance(self, odom_msg, model_states):
        robot_x, robot_y, _ = self._robot_xy_yaw(odom_msg)
        distances = []
        for name in self.ped_proxy_names:
            pose = model_states.get(name)
            if pose is None:
                continue
            distances.append(math.hypot(robot_x - pose.position.x, robot_y - pose.position.y))
        return float(min(distances)) if distances else 10.0

    def _build_teacher_state(self, scan_msg, odom_msg):
        raw_obs = {
            "ranges": np.asarray(scan_msg.ranges, dtype=np.float32),
            "prev_ranges": self.prev_scan_ranges,
            "dt": self.action_duration,
            "angle_min": scan_msg.angle_min,
            "angle_increment": scan_msg.angle_increment,
        }
        return extract_teacher_state(
            raw_obs=raw_obs,
            scan_angles=None,
            odom=odom_msg,
            goal=self.goal_xy,
            max_goal_distance=self.max_goal_distance,
            angle_config=self.teacher_angle_config,
        )

    def _sample_start_goal(self, seed=None):
        rng = np.random.RandomState(self.episode_seed if seed is None else int(seed))
        start_idx = int(rng.randint(0, len(self.start_points)))
        start = self.start_points[start_idx]

        start_arm = self.start_arm_ids[start_idx]
        if self.goal_indices_by_start_arm is not None:
            if start_arm is None:
                raise RuntimeError(
                    f"Scene '{self.scene_mode}' uses goal_indices_by_start_arm but start point {start_idx} has no arm id"
                )
            valid_goal_indices = list(self.goal_indices_by_start_arm.get(start_arm, ()))
            if not valid_goal_indices:
                raise RuntimeError(
                    f"Scene '{self.scene_mode}' has no explicit goal candidates for start arm '{start_arm}'"
                )
            goal_idx = valid_goal_indices[int(rng.randint(0, len(valid_goal_indices)))]
        elif start_arm is None or not any(arm is not None for arm in self.goal_arm_ids):
            goal_idx = int(rng.randint(0, len(self.goal_points)))
        else:
            valid_goal_indices = [idx for idx, arm in enumerate(self.goal_arm_ids) if arm != start_arm]
            if not valid_goal_indices:
                raise RuntimeError(f"No valid goal arms configured for scene '{self.scene_mode}'")
            goal_idx = valid_goal_indices[int(rng.randint(0, len(valid_goal_indices)))]

        goal = self.goal_points[goal_idx]
        return start, goal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ready, missing = self._wait_for_scene_ready()
        if not ready:
            raise RuntimeError(
                f"Scene '{self.scene_mode}' is not ready for training. Missing topics/models: "
                + ", ".join(missing)
            )
        self._stop_robot()
        self.reset_world()
        self._wall_sleep(0.5)

        start, goal = self._sample_start_goal(seed=seed)
        self.goal_xy = np.array(goal, dtype=np.float32)
        if not self._place_goal(goal[0], goal[1]):
            raise RuntimeError(f"Failed to place goal model '{self.goal_model_name}'")
        robot_set = self._set_robot_pose(*start)
        if not getattr(robot_set, "success", True):
            raise RuntimeError(f"Failed to set robot pose for '{self.robot_model_name}'")
        self._wall_sleep(0.5)
        self._stop_robot()
        self._wall_sleep(0.2)

        scan_msg, odom_msg, model_states = self._current_messages()
        if scan_msg is None or odom_msg is None:
            raise RuntimeError("Did not receive /scan and /odom after reset.")
        self.prev_scan_ranges = np.asarray(scan_msg.ranges, dtype=np.float32).copy() if scan_msg is not None else None
        self.prev_goal_distance = self._goal_distance(odom_msg)
        self.episode_min_human_distance = self._min_human_distance(odom_msg, model_states)
        self.steps = 0
        self.episode_seed = (self.episode_seed + 1) if seed is None else int(seed) + 1

        state = self._build_teacher_state(scan_msg, odom_msg)
        info = {
            "goal_xy": self.goal_xy.copy(),
            "start_pose": np.array(start, dtype=np.float32),
            "goal_distance": self.prev_goal_distance,
            "min_human_distance": self._min_human_distance(odom_msg, model_states),
            "episode_min_human_distance": self.episode_min_human_distance,
        }
        return state, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        repeat_steps = max(1, int(round(self.action_duration / max(self.dt, 1e-3))))
        for _ in range(repeat_steps):
            self.cmd_pub.publish(twist)
            self._wall_sleep(self.dt)

        self._stop_robot()
        self._wall_sleep(0.05)

        scan_msg, odom_msg, model_states = self._current_messages()
        if scan_msg is None or odom_msg is None:
            raise RuntimeError("Lost /scan or /odom during training step.")
        next_state = self._build_teacher_state(scan_msg, odom_msg)
        current_goal_distance = self._goal_distance(odom_msg)
        d_min_human = self._min_human_distance(odom_msg, model_states)
        self.episode_min_human_distance = min(float(self.episode_min_human_distance), float(d_min_human))
        min_scan = float(np.min(np.asarray(scan_msg.ranges, dtype=np.float32)))

        reached = current_goal_distance <= self.goal_radius
        collision = min_scan <= self.collision_distance
        self.steps += 1
        truncated = self.steps >= self.timeout_steps
        done = reached or collision

        r_goal = 0.0
        if collision:
            r_goal = self.collision_penalty
        elif reached:
            if self.episode_min_human_distance >= self.d_safe:
                r_goal = self.arrive_reward
            else:
                r_goal = self.arrive_reward - self.epsilon_1 * (self.d_safe - self.episode_min_human_distance)

        r_shaping = self.progress_scale * (self.prev_goal_distance - current_goal_distance)
        r_step = self.step_penalty
        reward = r_goal + r_shaping + r_step

        self.prev_goal_distance = current_goal_distance
        self.prev_scan_ranges = np.asarray(scan_msg.ranges, dtype=np.float32).copy()

        info = {
            "goal_distance": current_goal_distance,
            "min_human_distance": d_min_human,
            "episode_min_human_distance": self.episode_min_human_distance,
            "reached": reached,
            "collision": collision,
            "reward_goal": r_goal,
            "reward_shaping": r_shaping,
            "reward_step": r_step,
        }
        return next_state, float(reward), done, truncated, info
