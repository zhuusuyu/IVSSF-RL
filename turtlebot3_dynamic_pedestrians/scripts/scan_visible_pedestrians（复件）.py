#!/usr/bin/env python3

import math
import time

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Twist


def euler_to_quaternion(yaw):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class WaypointMover(object):
    def __init__(self):
        self.model_name = rospy.get_param("~model_name")
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.start_delay = float(rospy.get_param("~start_delay", 0.0))
        self.startup_delay = float(rospy.get_param("~startup_delay", 2.0))
        self.use_wall_time = bool(rospy.get_param("~use_wall_time", True))
        self.default_z = float(rospy.get_param("~default_z", 0.25))
        self.default_yaw = float(rospy.get_param("~default_yaw", 0.0))
        self.waypoints = self._load_waypoints(rospy.get_param("~waypoints", []))
        self.known_models = set()

        if len(self.waypoints) < 2:
            raise rospy.ROSInitException("~waypoints must contain at least two points")

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_model_states, queue_size=1)
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.t0_wall = time.time()
        self.t0_sim = rospy.Time.now().to_sec()
        rospy.loginfo(
            "[scan_visible_pedestrians] model=%s waypoints=%d start_delay=%.2f",
            self.model_name,
            len(self.waypoints),
            self.start_delay,
        )

    def _on_model_states(self, msg):
        self.known_models = set(msg.name)

    def _load_waypoints(self, raw_waypoints):
        waypoints = []
        for wp in raw_waypoints:
            if len(wp) < 3:
                continue
            t = float(wp[0])
            x = float(wp[1])
            y = float(wp[2])
            z = float(wp[3]) if len(wp) >= 4 else self.default_z
            yaw = float(wp[4]) if len(wp) >= 5 else self.default_yaw
            waypoints.append((t, x, y, z, yaw))
        return sorted(waypoints, key=lambda item: item[0])

    def _now(self):
        return time.time() if self.use_wall_time else rospy.Time.now().to_sec()

    def _elapsed(self):
        now = self._now()
        if self.use_wall_time:
            return now - self.t0_wall
        if now < self.t0_sim:
            self.t0_sim = now
        return now - self.t0_sim

    def _period(self):
        return max(self.waypoints[-1][0], 0.0)

    def sample_waypoint(self, elapsed):
        if elapsed < self.start_delay:
            _, x, y, z, yaw = self.waypoints[0]
            return x, y, z, yaw

        period = self._period()
        if period <= 0.0:
            _, x, y, z, yaw = self.waypoints[-1]
            return x, y, z, yaw

        t = (elapsed - self.start_delay) % period
        for idx in range(len(self.waypoints) - 1):
            t0, x0, y0, z0, yaw0 = self.waypoints[idx]
            t1, x1, y1, z1, yaw1 = self.waypoints[idx + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    return x1, y1, z1, yaw1
                s = (t - t0) / (t1 - t0)
                x = x0 + s * (x1 - x0)
                y = y0 + s * (y1 - y0)
                z = z0 + s * (z1 - z0)
                yaw = yaw0 + s * (yaw1 - yaw0)
                return x, y, z, yaw

        _, x, y, z, yaw = self.waypoints[0]
        return x, y, z, yaw

    def set_model_pose(self, x, y, z, yaw):
        if self.model_name not in self.known_models:
            rospy.logwarn_throttle(
                5.0,
                "[scan_visible_pedestrians] model %s not present in /gazebo/model_states yet",
                self.model_name,
            )
            return

        _, _, qz, qw = euler_to_quaternion(yaw)

        state = ModelState()
        state.model_name = self.model_name
        state.pose = Pose()
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.twist = Twist()
        state.reference_frame = "world"

        try:
            response = self.set_state(state)
            if not response.success:
                rospy.logwarn_throttle(
                    5.0,
                    "[scan_visible_pedestrians] set_model_state(%s) rejected: %s",
                    self.model_name,
                    response.status_message,
                )
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(
                5.0,
                "[scan_visible_pedestrians] set_model_state(%s) failed: %s",
                self.model_name,
                str(exc),
            )

    def run(self):
        if self.startup_delay > 0.0:
            time.sleep(self.startup_delay)

        wait_start = time.time()
        while not rospy.is_shutdown() and self.model_name not in self.known_models:
            if time.time() - wait_start > 15.0:
                rospy.logwarn(
                    "[scan_visible_pedestrians] timed out waiting for model %s; will keep retrying",
                    self.model_name,
                )
                break
            rospy.sleep(0.2)

        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            x, y, z, yaw = self.sample_waypoint(self._elapsed())
            self.set_model_pose(x, y, z, yaw)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("scan_visible_pedestrians")
    WaypointMover().run()
