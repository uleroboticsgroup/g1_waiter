"""Configuration classes for G1 humanoid robot training."""
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1RoughCfg(LeggedRobotCfg):
    """Configuration for G1 robot in rough terrain."""

    class init_state(LeggedRobotCfg.init_state):
        """Initial state configuration."""
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'torso_joint': 0.0
        }

    class env(LeggedRobotCfg.env):
        """Environment configuration."""
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class domain_rand(LeggedRobotCfg.domain_rand):
        """Domain randomization configuration."""
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class control(LeggedRobotCfg.control):
        """PD control configuration."""
        control_type = 'P'
        stiffness = {
            'hip_yaw': 100,
            'hip_roll': 100,
            'hip_pitch': 100,
            'knee': 150,
            'ankle': 40
        }
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 2
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        """Robot asset configuration."""
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        """Reward configuration."""
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales(LeggedRobotCfg.rewards.scales):
            """Reward scales."""
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18


class G1RoughCfgPPO(LeggedRobotCfgPPO):
    """PPO configuration for G1 robot training."""

    class policy:
        """Policy network configuration."""
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        """PPO algorithm configuration."""
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        """Training runner configuration."""
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'
