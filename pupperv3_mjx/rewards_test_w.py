import jax
from jax import numpy as jp
from brax.base import Motion, Transform
from brax import base, math
import numpy as np

EPS = 1e-6
# ------------ reward functions----------------
def reward_lin_vel_z(xd: Motion) -> jax.Array:
    """Penalize vertical base movement to maintain stable height.
    
    Args:
        xd: Motion state containing velocity information
    Returns:
        Negative reward proportional to the square of z-axis velocity
    """
    return jp.clip(jp.square(xd.vel[0, 2]), -1000.0, 1000.0)


def reward_ang_vel_xy(xd: Motion) -> jax.Array:
    """Penalize tilting motion to maintain stable orientation.
    
    Args:
        xd: Motion state containing angular velocity information
    Returns:
        Negative reward based on x and y angular velocities
    """
    return jp.clip(jp.sum(jp.square(xd.ang[0, :2])), -1000.0, 1000.0)


def reward_tracking_orientation(
    desired_world_z_in_body_frame: jax.Array, x: Transform, tracking_sigma: float
) -> jax.Array:
    """Reward for maintaining desired body orientation.
    
    Args:
        desired_world_z_in_body_frame: Target z-axis in body frame
        x: Current transform state
        tracking_sigma: Controls how strictly to track the orientation
    Returns:
        Reward based on how closely current orientation matches desired
    """
    world_z = jp.array([0.0, 0.0, 1.0])
    world_z_in_body_frame = math.rotate(world_z, math.quat_inv(x.rot[0]))
    error = jp.sum(jp.square(world_z_in_body_frame - desired_world_z_in_body_frame))
    return jp.clip(jp.exp(-error / (tracking_sigma + EPS)), -1000.0, 1000.0)


def reward_orientation(x: Transform) -> jax.Array:
    """Penalize deviation from upright orientation.
    
    Args:
        x: Current transform state
    Returns:
        Negative reward based on tilt from vertical
    """
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.clip(jp.sum(jp.square(rot_up[:2])), -1000.0, 1000.0)


def reward_torques(torques: jax.Array) -> jax.Array:
    """Penalize high joint torques to encourage energy efficiency.
    
    Args:
        torques: Array of joint torques
    Returns:
        Negative reward proportional to sum of squared torques
    """
    return jp.clip(jp.sum(jp.square(torques)), -1000.0, 1000.0)


def reward_joint_acceleration(
    joint_vel: jax.Array, last_joint_vel: jax.Array, dt: float
) -> jax.Array:
    """Penalize high joint accelerations for smoother motion.
    
    Args:
        joint_vel: Current joint velocities
        last_joint_vel: Previous joint velocities
        dt: Time step
    Returns:
        Negative reward based on sum of squared joint accelerations
    """
    return jp.clip(jp.sum(jp.square((joint_vel - last_joint_vel) / (dt + EPS))), -1000.0, 1000.0)


def reward_mechanical_work(torques: jax.Array, velocities: jax.Array) -> jax.Array:
    """Penalize mechanical work to encourage energy efficiency.
    
    Args:
        torques: Array of joint torques
        velocities: Array of joint velocities
    Returns:
        Negative reward proportional to mechanical work
    """
    return jp.clip(jp.sum(jp.abs(torques * velocities)), -1000.0, 1000.0)


def reward_action_rate(act: jax.Array, last_act: jax.Array) -> jax.Array:
    """Penalize rapid changes in actions for smoother control.
    
    Args:
        act: Current action vector
        last_act: Previous action vector
    Returns:
        Negative reward based on squared difference between actions
    """
    return jp.clip(jp.sum(jp.square(act - last_act)), -1000.0, 1000.0)


def reward_tracking_lin_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    """Reward for tracking desired linear velocity commands.
    
    Args:
        commands: Desired velocity commands [vx, vy, ωz]
        x: Current transform state
        xd: Current motion state
        tracking_sigma: Controls tracking strictness
    Returns:
        Reward based on how well linear velocity is tracked
    """
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(-lin_vel_error / (tracking_sigma + EPS))
    return jp.clip(lin_vel_reward, -1000.0, 1000.0)


def reward_tracking_ang_vel(
    commands: jax.Array, x: Transform, xd: Motion, tracking_sigma
) -> jax.Array:
    """Reward for tracking desired angular velocity (yaw) commands.
    
    Args:
        commands: Desired velocity commands [vx, vy, ωz]
        x: Current transform state
        xd: Current motion state
        tracking_sigma: Controls tracking strictness
    Returns:
        Reward based on how well angular velocity is tracked
    """
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.clip(jp.exp(-ang_vel_error / (tracking_sigma + EPS)), -1000.0, 1000.0)


def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    commands: jax.Array,
    minimum_airtime: float = 0.1,
) -> jax.Array:
    """Reward for maintaining appropriate foot air time during locomotion.
    
    Args:
        air_time: Time since each foot was last in contact
        first_contact: Binary mask for first contact after being in air
        commands: Current motion commands
        minimum_airtime: Minimum desired air time for reward
    Returns:
        Reward based on how well foot air time matches desired
    """
    rew_air_time = jp.sum((air_time - minimum_airtime) * first_contact)
    rew_air_time *= math.normalize(commands[:3])[1] > 0.05  # no reward for zero command
    return jp.clip(rew_air_time, -1000.0, 1000.0)


def reward_abduction_angle(
    joint_angles: jax.Array, desired_abduction_angles: jax.Array = jp.zeros(4)
):
    """Penalize deviation of leg abduction angles from desired values.
    
    Args:
        joint_angles: Current joint angles
        desired_abduction_angles: Target abduction angles for each leg
    Returns:
        Negative reward based on squared error from desired abduction angles
    """
    return jp.clip(jp.sum(jp.square(joint_angles[1::3] - desired_abduction_angles)), -1000.0, 1000.0)


def reward_stand_still(
    commands: jax.Array,
    joint_angles: jax.Array,
    default_pose: jax.Array,
    command_threshold: float,
) -> jax.Array:
    """Penalize movement when robot should be standing still.
    
    When velocity commands are below threshold, encourages maintaining default pose.
    
    Args:
        commands: Robot velocity commands [vx, vy, ωz]
        joint_angles: Current joint angles
        default_pose: Default joint angles for standing
        command_threshold: Velocity command norm threshold below which to penalize movement
    Returns:
        Negative reward when moving without command, zero otherwise
    """
    return jp.clip(
        jp.sum(jp.abs(joint_angles - default_pose)) * (
            math.normalize(commands[:3])[1] < command_threshold
        ),
        -1000.0,
        1000.0
    )


def reward_foot_slip(
    pipeline_state: base.State,
    contact_filt: jax.Array,
    feet_site_id: np.array,
    lower_leg_body_id: np.array,
) -> jax.Array:
    """Penalize foot sliding when in contact with the ground.
    
    Args:
        pipeline_state: Current physics simulation state
        contact_filt: Binary mask indicating which feet are in contact
        feet_site_id: Indices for foot sites in the simulation
        lower_leg_body_id: Indices for lower leg bodies
    Returns:
        Negative reward proportional to foot sliding velocity when in contact
    """
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.clip(
        jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1))),
        -1000.0,
        1000.0
    )


def reward_termination(done: jax.Array, step: jax.Array, step_threshold: int) -> jax.Array:
    """Terminate episode if robot falls or fails early.
    
    Args:
        done: Whether the episode is done
        step: Current timestep in episode
        step_threshold: Minimum timesteps before allowing termination
    Returns:
        Boolean indicating if episode should terminate
    """
    return done & (step < step_threshold)


def reward_geom_collision(pipeline_state: base.State, geom_ids: np.array) -> jax.Array:
    """Penalize unwanted collisions between specified geometries.
    
    Args:
        pipeline_state: Current physics simulation state
        geom_ids: List of geometry IDs to check for collisions
    Returns:
        Negative reward for each unwanted collision
    """
    contact = jp.array(0.0)
    for id in geom_ids:
        contact += jp.sum(
            ((pipeline_state.contact.geom1 == id) | (pipeline_state.contact.geom2 == id))
            * (pipeline_state.contact.dist < 0.0)
        )
    return jp.clip(contact, -1000.0, 1000.0)
