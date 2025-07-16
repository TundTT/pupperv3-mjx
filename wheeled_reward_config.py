reward_config = config_dict.ConfigDict()
reward_config.rewards = config_dict.ConfigDict()
reward_config.rewards.scales = config_dict.ConfigDict()

# === PRIMARY TRACKING REWARDS (High Priority) ===
# Track linear velocity - increased for precise wheeled control
reward_config.rewards.scales.tracking_lin_vel = 2.5

# Track angular velocity (yaw) - increased for turning precision
reward_config.rewards.scales.tracking_ang_vel = 1.5

# === STABILITY REWARDS (Critical for wheeled robots) ===
# Penalize vertical movement - wheels shouldn't bounce
reward_config.rewards.scales.lin_vel_z = -5.0

# Penalize roll/pitch rates - prevent tipping over
reward_config.rewards.scales.ang_vel_xy = -2.0

# Penalize roll/pitch angles - keep upright
reward_config.rewards.scales.orientation = -8.0

# === CONTROL SMOOTHNESS ===
# Smooth actions for stable wheeled motion
reward_config.rewards.scales.action_rate = -0.05

# Motor efficiency
reward_config.rewards.scales.torques = -0.001

# === STATIONARY BEHAVIOR ===
# Maintain position when no commands given
reward_config.rewards.scales.stand_still = -1.0

# === COLLISION AVOIDANCE ===
# Prevent body collisions
reward_config.rewards.scales.body_collision = -10.0

# Early termination penalty
reward_config.rewards.scales.termination = -200.0

# === DISABLED REWARDS (Not applicable for wheeled motion) ===
# Remove walking-specific rewards
reward_config.rewards.scales.tracking_orientation = 0  # Not needed for wheels
reward_config.rewards.scales.joint_acceleration = 0   # Less relevant for wheels
reward_config.rewards.scales.mechanical_work = 0      # Not critical
reward_config.rewards.scales.feet_air_time = 0        # No feet for wheels
reward_config.rewards.scales.stand_still_joint_velocity = 0  # Simplified
reward_config.rewards.scales.abduction_angle = 0      # No leg abduction
reward_config.rewards.scales.foot_slip = 0            # Could adapt for wheel slip
reward_config.rewards.scales.knee_collision = 0       # No knees

# Tracking reward parameters
reward_config.rewards.tracking_sigma = 0.25