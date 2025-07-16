import jax
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math
import numpy as np

EPS = 1e-6
# ------------ reward functions----------------

# Base Linear Velocity Z-Axis Penalty
# Discourages vertical movement of the robot base
def reward_lin_vel_z(xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    # Normalize to 0-1 range assuming max reasonable z-vel is ~2 m/s
    return jp.clip(jp.square(xd.vel[0, 2]) / 4.0, 0.0, 1.0)


# Base Angular Velocity XY-Axes Penalty
# Discourages roll and pitch rotations of the robot base
def reward_ang_vel_xy(xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    # Normalize to 0-1 range assuming max reasonable angular vel is ~5 rad/s
    return jp.clip(jp.sum(jp.square(xd.ang[0, :2])) / 25.0, 0.0, 1.0)


# Desired Orientation Tracking Reward
# Rewards the robot for maintaining a desired body orientation
def reward_tracking_orientation(
    desired_world_z_in_body_frame: jax.Array, x: Transform, tracking_sigma: float
) -> jax.Array:
    # Tracking of desired body orientation
    world_z = jp.array([0.0, 0.0, 1.0])
    world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(world_z_in_body_frame - desired_world_z_in_body_frame))
    # Already normalized to 0-1 range via exponential
    return jp.clip(jp.exp(-error / (tracking_sigma + EPS)), 0.0, 1.0)


# Base Orientation Penalty
# Penalizes deviation from upright (flat) base orientation
def reward_orientation(x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    # Normalize to 0-1 range (max deviation would be ~2.0 for completely inverted)
    return jp.clip(jp.sum(jp.square(rot_up[:2])) / 2.0, 0.0, 1.0)


# Joint Torque Penalty
# Encourages energy-efficient movement by penalizing high torques
def reward_torques(torques: jax.Array) -> jax.Array:
    # Penalize torques
    # Normalize to 0-1 range assuming max reasonable torque is ~10 Nm per joint
    # For 12 joints, max sum of squares would be ~1200
    return jp.clip(jp.sum(jp.square(torques)) / 1200.0, 0.0, 1.0)


# Joint Acceleration Penalty
# Promotes smooth joint motion by penalizing rapid velocity changes
def reward_joint_acceleration(
    joint_vel: jax.Array, last_joint_vel: jax.Array, dt: float
) -> jax.Array:
    # Normalize to 0-1 range assuming max reasonable acceleration is ~100 rad/s²
    # For 12 joints, max sum of squares would be ~120000
    acceleration = jp.sum(jp.square((joint_vel - last_joint_vel) / (dt + EPS)))
    return jp.clip(acceleration / 120000.0, 0.0, 1.0)


# Mechanical Work Penalty
# Encourages energy efficiency by penalizing mechanical work (power consumption)
def reward_mechanical_work(torques: jax.Array, velocities: jax.Array) -> jax.Array:
    # Penalize mechanical work
    # Normalize to 0-1 range assuming max power is ~50W per joint (10Nm * 5rad/s)
    # For 12 joints, max total would be ~600W
    return jp.clip(jp.sum(jp.abs(torques * velocities)) / 600.0, 0.0, 1.0)


# Action Rate Penalty
# Promotes smooth control by penalizing rapid changes in actions
def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Penalize changes in actions
    # Normalize to 0-1 range assuming max action change is ~2.0 per joint
    # For 12 joints, max sum of squares would be ~48
    return jp.clip(jp.sum(jp.square(act - last_act)) / 48.0, 0.0, 1.0)


# Linear Velocity Command Tracking Reward
# Rewards the robot for following desired linear velocity commands (forward/backward, left/right)
def reward_tracking_lin_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(-lin_vel_error / (tracking_sigma + EPS))
    # Already normalized to 0-1 range via exponential
    return jp.clip(lin_vel_reward, 0.0, 1.0)


# Angular Velocity Command Tracking Reward
# Rewards the robot for following desired yaw rotation commands
def reward_tracking_ang_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    # Already normalized to 0-1 range via exponential
    return jp.clip(jp.exp(-ang_vel_error / (tracking_sigma + EPS)), 0.0, 1.0)


# Feet Air Time Reward
# Encourages proper gait by rewarding feet that stay airborne for adequate time during locomotion
def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    minimum_airtime: float = 0.1,
) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - minimum_airtime) * first_contact)
    rew_air_time *= math.normalize(commands[:3])[1] > 0.05  # no reward for zero command
    # Normalize to 0-1 range assuming max air time benefit is ~0.5s for 4 feet
    return jp.clip(rew_air_time / 2.0, 0.0, 1.0)


# Hip Abduction Angle Penalty
# Maintains proper leg stance by penalizing deviation from desired hip abduction angles
def reward_abduction_angle(
    joint_angles: jax.Array, desired_abduction_angles: jax.Array = jp.zeros(4)
):
    # Penalize abduction angle
    # Normalize to 0-1 range assuming max abduction deviation is ~π/2 rad per joint
    # For 4 hip joints, max sum of squares would be ~π²
    return jp.clip(jp.sum(jp.square(joint_angles[1::3] - desired_abduction_angles)) / (jp.pi**2), 0.0, 1.0)


# Stand Still Penalty
# Encourages the robot to maintain default pose when no movement commands are given
def reward_stand_still(
    commands: jax.Array,
    joint_angles: jax.Array,
    default_pose: jax.Array,
    command_threshold: float,
) -> jax.Array:
    """
    Penalize motion at zero commands
    Args:
        commands: robot velocity commands
        joint_angles: joint angles
        default_pose: default pose
        command_threshold: if norm of commands is less than this, return non-zero penalty
    """

    # Penalize motion at zero commands
    # Normalize to 0-1 range assuming max joint deviation is ~π rad per joint
    # For 12 joints, max sum of abs would be ~12π
    penalty = jp.sum(jp.abs(joint_angles - default_pose)) * (
        math.normalize(commands[:3])[1] < command_threshold
    )
    return jp.clip(penalty / (12.0 * jp.pi), 0.0, 1.0)


# Foot Slip Penalty
# Prevents foot slipping by penalizing high foot velocities when feet are in contact with ground
def reward_foot_slip(
    pipeline_state: base.State,
    contact_filt: jax.Array,
    feet_site_id: np.array,
    lower_leg_body_id: np.array,
) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
    # Penalize large feet velocity for feet that are in contact with the ground.
    # Normalize to 0-1 range assuming max foot slip velocity is ~2 m/s
    # For 4 feet with 2 velocity components each, max sum of squares would be ~16
    slip_penalty = jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))
    return jp.clip(slip_penalty / 16.0, 0.0, 1.0)


# Early Termination Penalty
# Penalizes episodes that terminate early (before reaching step threshold)
def reward_termination(done: jax.Array, step: jax.Array, step_threshold: int) -> jax.Array:
    return done & (step < step_threshold)


# Geometry Collision Penalty
# Penalizes collisions between specified geometry bodies to prevent self-collision or unwanted contacts
def reward_geom_collision(pipeline_state: base.State, geom_ids: np.array) -> jax.Array:
    contact = jp.array(0.0)
    for id in geom_ids:
        contact += jp.sum(
            ((pipeline_state.contact.geom1 == id) | (pipeline_state.contact.geom2 == id))
            * (pipeline_state.contact.dist < 0.0)
        )
    # Normalize to 0-1 range assuming max reasonable collision count is ~10
    return jp.clip(contact / 10.0, 0.0, 1.0)
