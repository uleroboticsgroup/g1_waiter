"""MuJoCo deployment script for Unitree robots with trained RL policies."""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion):
    """Compute gravity vector in body frame from quaternion.

    Args:
        quaternion: Base orientation as [w, x, y, z].

    Returns:
        Gravity vector in body frame.
    """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Compute PD control torques.

    Args:
        target_q: Target joint positions.
        q: Current joint positions.
        kp: Proportional gains.
        target_dq: Target joint velocities.
        dq: Current joint velocities.
        kd: Derivative gains.

    Returns:
        Joint torques.
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def load_config(config_file):
    """Load configuration from YAML file.

    Args:
        config_file: Name of config file in configs folder.

    Returns:
        Configuration dictionary.
    """
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}"
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    """Run MuJoCo simulation with trained policy."""
    parser = argparse.ArgumentParser(description="Deploy trained policy in MuJoCo")
    parser.add_argument("config_file", type=str, help="Config file name in configs folder")
    args = parser.parse_args()

    config = load_config(args.config_file)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)
    policy.eval()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            tau = pd_control(
                target_dof_pos, d.qpos[7:], kps,
                np.zeros_like(kds), d.qvel[6:], kds
            )
            d.ctrl[:] = tau

            mujoco.mj_step(m, d)
            counter += 1

            if counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9 + num_actions] = qj
                obs[9 + num_actions:9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action = policy(obs_tensor).numpy().squeeze()

                target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()