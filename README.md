# G1 Waiter - Humanoid Robot with Reinforcement Learning

<div align="center">

### Demo Video

[![G1 Robot Demo](https://img.shields.io/badge/▶%20Watch%20Video-Google%20Drive-red?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1-MNJGQ7xh17Pq5iRw3uyyD1lHApfhsYp/view?usp=drivesdk)

</div>

---

## Description

Bipedal locomotion training for the Unitree G1 humanoid robot using Deep Reinforcement Learning (Deep RL) with the PPO algorithm and recurrent LSTM networks.

---



## Simulation Environment

The `scene.xml` file defines a terrain with random obstacles (bumps) of cylinder and box types distributed across the simulation area to train the robot's robustness on irregular terrain.

---

## Project Structure

```
G1_waiter/
├── README.md
├── config/
│   └── g1_config.py      # Training configuration
├── env/
│   └── g1_env.py         # Custom environment
├── scripts/
│   └── train.py          # Training script
├── deploy/
│   ├── deploy_mujoco.py  # MuJoCo deployment
│   ├── g1.yaml           # Deployment config
│   └── scene.xml         # MuJoCo scene
└── models/
    └── policy_lstm_1.pt  # Trained model
```

---

## Project Files

### `config/g1_config.py` - Training Configuration
Defines all hyperparameters and settings for training the G1 robot. Contains two main classes:
- **G1RoughCfg**: Robot physical configuration including initial pose, joint angles, PD control gains, reward function weights, domain randomization parameters, and contact/collision settings.
- **G1RoughCfgPPO**: PPO algorithm configuration including neural network architecture (LSTM with 64 hidden units), actor-critic hidden layers, and training parameters.

### `env/g1_env.py` - Custom Environment
Implements the G1 robot environment by extending `LeggedRobot` from legged_gym. Key features:
- Custom observation computation with gait phase encoding (sin/cos)
- Foot state tracking for contact detection
- Custom reward functions: `_reward_contact` (correct foot timing), `_reward_feet_swing_height` (target swing height), `_reward_contact_no_vel` (penalize sliding), `_reward_hip_pos` (hip alignment)

### `scripts/train.py` - Training Script
Main training script with PPO algorithm and early stopping mechanism. Monitors mean reward during training and stops if no improvement is detected after 100 iterations (patience), saving the best model automatically.

### `deploy/deploy_mujoco.py` - MuJoCo Deployment
Loads a trained policy (.pt file) and runs it in the MuJoCo physics simulator. Implements:
- PD control loop at 500 Hz (0.002s timestep)
- Observation construction matching training format
- Real-time visualization with MuJoCo viewer

### `deploy/g1.yaml` - Deployment Configuration
YAML file containing all parameters needed for MuJoCo deployment: policy path, simulation settings, PD gains (Kp/Kd), default joint angles, observation scales, and initial velocity commands.

### `deploy/scene.xml` - MuJoCo Scene
MuJoCo XML scene definition including the G1 robot model (`g1_12dof.xml`), ground plane, lighting, and randomly distributed obstacles (cylinders and boxes) to test robustness on uneven terrain.

### `models/policy_lstm_1.pt` - Trained Model
PyTorch JIT-compiled LSTM policy network trained with PPO. Takes 47-dimensional observations as input and outputs 12 joint position targets.

---
## System Architecture

### Robot: Unitree G1
- **Degrees of Freedom (DoF):** 12 active joints
- **Joints per leg:** 6 (hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
- **Target height:** 0.78 m
- **Initial position:** [0.0, 0.0, 0.8] m

### Neural Network (Policy)
| Parameter | Value |
|-----------|-------|
| Policy type | ActorCriticRecurrent |
| RNN architecture | LSTM |
| Actor hidden layers | [32] |
| Critic hidden layers | [32] |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| Activation function | ELU |
| Initial noise | 0.8 |

### Observation and Action Space
| Component | Dimension |
|-----------|-----------|
| Observations | 47 |
| Privileged observations | 50 |
| Actions | 12 |

**Observation vector (47 dim):**
- Base angular velocity (3)
- Projected gravity vector (3)
- Velocity commands (3)
- Joint positions (12)
- Joint velocities (12)
- Previous actions (12)
- Gait phase (sin, cos) (2)

---

## PD Control

### Stiffness Gains (Kp)
| Joint | Value |
|-------|-------|
| hip_yaw | 100 |
| hip_roll | 100 |
| hip_pitch | 100 |
| knee | 150 |
| ankle | 40 |

### Damping Gains (Kd)
| Joint | Value |
|-------|-------|
| hip_yaw | 2 |
| hip_roll | 2 |
| hip_pitch | 2 |
| knee | 4 |
| ankle | 2 |

### Control Parameters
- **Action scale:** 0.25
- **Decimation:** 4

---

## Reward Function

| Reward | Scale | Description |
|--------|-------|-------------|
| tracking_lin_vel | 1.0 | Linear velocity tracking |
| tracking_ang_vel | 0.5 | Angular velocity tracking |
| lin_vel_z | -2.0 | Vertical velocity penalty |
| ang_vel_xy | -0.05 | Lateral angular velocity penalty |
| orientation | -1.0 | Incorrect orientation penalty |
| base_height | -10.0 | Base height penalty |
| dof_acc | -2.5e-7 | Joint acceleration penalty |
| dof_vel | -1e-3 | Joint velocity penalty |
| action_rate | -0.01 | Action change penalty |
| dof_pos_limits | -5.0 | Joint limits penalty |
| alive | 0.15 | Reward for staying active |
| hip_pos | -1.0 | Hip position penalty |
| contact_no_vel | -0.2 | Feet sliding penalty |
| feet_swing_height | -20.0 | Swing height penalty |
| contact | 0.18 | Correct contact reward |

---

## Domain Randomization

| Parameter | Range/Value |
|-----------|-------------|
| Friction | [0.1, 1.25] |
| Additional base mass | [-1.0, 3.0] kg |
| External pushes | Enabled |
| Push interval | 5 s |
| Maximum push velocity | 1.5 m/s |

---

## Training

### PPO Algorithm
| Parameter | Value |
|-----------|-------|
| Entropy coefficient | 0.01 |
| Maximum iterations | 10,000 |

### Early Stopping
| Parameter | Value |
|-----------|-------|
| Patience | 100 iterations |
| Minimum delta | 0.001 |

---

## MuJoCo Simulation

### Simulation Parameters
| Parameter | Value |
|-----------|-------|
| Duration | 60 s |
| Time step | 0.002 s |
| Control decimation | 10 |
| Gait period | 0.8 s |

### Observation Scales
| Parameter | Scale |
|-----------|-------|
| Angular velocity | 0.25 |
| Joint position | 1.0 |
| Joint velocity | 0.05 |
| Commands | [2.0, 2.0, 0.25] |

### Initial Command
- **Linear velocity X:** 0.5 m/s
- **Linear velocity Y:** 0.0 m/s
- **Angular velocity Z:** 0.0 rad/s

---

## Default Angles (rad)

| Joint | Left | Right |
|-------|------|-------|
| hip_pitch | -0.1 | -0.1 |
| hip_roll | 0.0 | 0.0 |
| hip_yaw | 0.0 | 0.0 |
| knee | 0.3 | 0.3 |
| ankle_pitch | -0.2 | -0.2 |
| ankle_roll | 0.0 | 0.0 |

---
## Dependencies

- Isaac Gym (NVIDIA)
- legged_gym
- MuJoCo
- PyTorch
- NumPy

---

## Execution

### Training
```bash
python scripts/train.py --task g1
```

### MuJoCo Deployment
```bash
python deploy/deploy_mujoco.py deploy/g1.yaml
```

---

## Acknowledgments

This work has been funded by:

- **CENTAURO** - TransMisiones 2023 Application 162789
- **SWEET** - Marie Skłodowska-Curie Actions, Grant Agreement No. 101168792
- **EXPLICIT** - PID2024-162298OB-I00

---

## References

- **unitree_rl_gym**: https://github.com/unitreerobotics/unitree_rl_gym
