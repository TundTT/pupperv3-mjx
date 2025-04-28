from ml_collections import config_dict


def get_config():
    """Returns reward config for barkour quadruped environment.
       I have set all rewards to 0.0 to disable you from looking at the answers!
    """

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=0.0,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.0,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=0.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=0.0,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=0.0,
                        # Track desired body orientation
                        tracking_orientation=0.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=0.0,
                        # L2 reg of joint acceleration
                        joint_acceleration=0.0,
                        # L1 regularization of mechanical work, |v * tau|.
                        mechanical_work=0.0,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=0.0,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.0,
                        # Encourage no motion at zero command, L1 regularization
                        # |q - q_default|.
                        stand_still=0.0,
                        # Encourage no motion at zero command, L1 regularization
                        # |q_dot|.
                        stand_still_joint_velocity=0.0,
                        # Abduction angle penalty
                        abduction_angle=0.0,
                        # Early termination penalty.
                        termination=0.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=0.0,
                        # Penalize knee hitting ground
                        knee_collision=0.0,
                        # Penalize body hitting ground,
                        body_collision=0.0,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.0,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config
