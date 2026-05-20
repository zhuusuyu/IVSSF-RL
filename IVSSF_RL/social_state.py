import math

import numpy as np


DEFAULT_SOCIAL_ANGLE_CONFIG = {
    "front": ((340.0, 360.0), (0.0, 20.0)),
    "left": ((20.0, 100.0),),
    "right": ((260.0, 340.0),),
}


def convert_angle_to_social_deg(raw_angle_rad):
    return float(np.degrees(raw_angle_rad) % 360.0)


def build_social_angle_array(scan_angles_rad):
    scan_angles_rad = np.asarray(scan_angles_rad, dtype=np.float32)
    return np.mod(np.degrees(scan_angles_rad), 360.0)


def build_scan_angles_from_metadata(angle_min, angle_increment, num_ranges):
    indices = np.arange(int(num_ranges), dtype=np.float32)
    return float(angle_min) + indices * float(angle_increment)


def make_sector_mask(social_angles_deg, intervals):
    mask = np.zeros_like(social_angles_deg, dtype=bool)
    for start_deg, end_deg in intervals:
        if start_deg <= end_deg:
            mask |= (social_angles_deg >= start_deg) & (social_angles_deg <= end_deg)
        else:
            mask |= (social_angles_deg >= start_deg) | (social_angles_deg <= end_deg)
    return mask


def sanitize_ranges(ranges, range_max=6.0):
    ranges = np.asarray(ranges, dtype=np.float32)
    ranges = np.where(np.isfinite(ranges), ranges, range_max)
    return np.clip(ranges, 0.0, range_max)


def compute_range_rates(ranges, prev_ranges=None, dt=0.1):
    ranges = np.asarray(ranges, dtype=np.float32)
    if prev_ranges is None or dt is None or dt <= 1e-6:
        return np.zeros_like(ranges, dtype=np.float32)
    prev_ranges = np.asarray(prev_ranges, dtype=np.float32)
    prev_ranges = np.where(np.isfinite(prev_ranges), prev_ranges, ranges)
    return (prev_ranges - ranges) / float(dt)


def compute_social_pressure(
    ranges,
    range_rates,
    sector_mask,
    distance_safe=1.6,
    occupancy_threshold=1.2,
    approach_ref=0.8,
):
    ranges = sanitize_ranges(ranges, range_max=max(distance_safe, occupancy_threshold, 6.0))
    range_rates = np.asarray(range_rates, dtype=np.float32)
    if range_rates.shape != ranges.shape:
        range_rates = np.zeros_like(ranges, dtype=np.float32)

    sector_mask = np.asarray(sector_mask, dtype=bool)
    if sector_mask.shape != ranges.shape or not np.any(sector_mask):
        return 0.0

    sector_ranges = ranges[sector_mask]
    sector_rates = range_rates[sector_mask]

    min_distance = float(np.min(sector_ranges))
    distance_risk = np.clip((distance_safe - min_distance) / max(distance_safe, 1e-6), 0.0, 1.0)

    approach_speed = float(np.max(np.maximum(sector_rates, 0.0)))
    approach_risk = np.clip(approach_speed / max(approach_ref, 1e-6), 0.0, 1.0)

    occupancy_risk = float(np.mean(sector_ranges < occupancy_threshold))

    pressure = 0.5 * distance_risk + 0.3 * approach_risk + 0.2 * occupancy_risk
    return float(np.clip(pressure, 0.0, 1.0))


def _normalize_goal_distance(distance, max_goal_distance):
    return float(np.clip(distance / max(max_goal_distance, 1e-6), 0.0, 1.0))


def _normalize_heading(relative_heading_rad):
    wrapped = math.atan2(math.sin(relative_heading_rad), math.cos(relative_heading_rad))
    return float(np.clip(wrapped / math.pi, -1.0, 1.0))


def _extract_xy_yaw_from_odom(odom):
    if odom is None:
        return 0.0, 0.0, 0.0

    if isinstance(odom, dict):
        if "yaw" in odom:
            return float(odom.get("x", 0.0)), float(odom.get("y", 0.0)), float(odom["yaw"])
        orientation = odom.get("orientation", {})
        z_val = float(orientation.get("z", 0.0))
        w_val = float(orientation.get("w", 1.0))
        yaw = math.atan2(2.0 * w_val * z_val, 1.0 - 2.0 * z_val * z_val)
        return float(odom.get("x", 0.0)), float(odom.get("y", 0.0)), yaw

    pose = odom.pose.pose if hasattr(odom, "pose") and hasattr(odom.pose, "pose") else odom.pose
    pos = pose.position
    ori = pose.orientation
    yaw = math.atan2(2.0 * (ori.w * ori.z + ori.x * ori.y), 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z))
    return float(pos.x), float(pos.y), float(yaw)


def extract_teacher_state(
    raw_obs,
    scan_angles,
    odom,
    goal,
    *,
    max_goal_distance=6.0,
    front_clearance_ref=2.5,
    distance_safe=1.6,
    occupancy_threshold=1.2,
    approach_ref=0.8,
    angle_config=None,
):
    angle_config = angle_config or DEFAULT_SOCIAL_ANGLE_CONFIG

    if isinstance(raw_obs, dict):
        ranges = raw_obs.get("ranges")
        prev_ranges = raw_obs.get("prev_ranges")
        dt = raw_obs.get("dt", 0.1)
        if scan_angles is None and "angle_min" in raw_obs and "angle_increment" in raw_obs:
            scan_angles = build_scan_angles_from_metadata(
                raw_obs["angle_min"],
                raw_obs["angle_increment"],
                len(ranges),
            )
    else:
        ranges = raw_obs
        prev_ranges = None
        dt = 0.1

    ranges = sanitize_ranges(ranges, range_max=max_goal_distance)
    range_rates = compute_range_rates(ranges, prev_ranges=prev_ranges, dt=dt)
    social_angles_deg = build_social_angle_array(scan_angles)

    front_mask = make_sector_mask(social_angles_deg, angle_config["front"])
    left_mask = make_sector_mask(social_angles_deg, angle_config["left"])
    right_mask = make_sector_mask(social_angles_deg, angle_config["right"])

    front_clearance = float(np.min(ranges[front_mask])) if np.any(front_mask) else front_clearance_ref
    front_clearance = float(np.clip(front_clearance / max(front_clearance_ref, 1e-6), 0.0, 1.0))

    left_pressure = compute_social_pressure(
        ranges,
        range_rates,
        left_mask,
        distance_safe=distance_safe,
        occupancy_threshold=occupancy_threshold,
        approach_ref=approach_ref,
    )
    right_pressure = compute_social_pressure(
        ranges,
        range_rates,
        right_mask,
        distance_safe=distance_safe,
        occupancy_threshold=occupancy_threshold,
        approach_ref=approach_ref,
    )

    robot_x, robot_y, robot_yaw = _extract_xy_yaw_from_odom(odom)
    goal_x, goal_y = float(goal[0]), float(goal[1])
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    goal_distance = _normalize_goal_distance(math.hypot(dx, dy), max_goal_distance=max_goal_distance)
    goal_heading = _normalize_heading(math.atan2(dy, dx) - robot_yaw)

    return np.array(
        [goal_distance, goal_heading, front_clearance, left_pressure, right_pressure],
        dtype=np.float32,
    )
