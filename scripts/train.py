"""Training script with early stopping for legged robots."""
import os
import sys
import statistics
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

EARLY_STOPPING_ENABLED = True
PATIENCE = 100
MIN_DELTA = 0.001


def train(args):
    """Train a policy with optional early stopping.
    
    Args:
        args: Command line arguments from get_args().
    """
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    if EARLY_STOPPING_ENABLED:
        best_reward = None
        no_improve_count = 0
        original_log = ppo_runner.log
        
        def log_with_early_stop(locs, width=80, pad=35):
            nonlocal best_reward, no_improve_count
            original_log(locs, width, pad)
            
            mean_reward = locs.get('mean_reward')
            if mean_reward is None:
                mean_reward = locs.get('ep_rew_mean')
            if mean_reward is None:
                rewbuffer = locs.get('rewbuffer', [0])
                if hasattr(rewbuffer, '__len__') and len(rewbuffer) > 0:
                    mean_reward = statistics.mean(rewbuffer)
                else:
                    mean_reward = 0
            
            iteration = locs.get('it', 0)
            
            if best_reward is None:
                best_reward = mean_reward
                print(f"  [Early Stop] Initial: {best_reward:.4f}")
                return
            
            if mean_reward > best_reward + MIN_DELTA:
                best_reward = mean_reward
                no_improve_count = 0
                print(f"  [Early Stop] New best: {best_reward:.4f}")
            else:
                no_improve_count += 1
                if no_improve_count % 10 == 0:
                    print(f"  [Early Stop] {no_improve_count}/{PATIENCE} (best: {best_reward:.4f}, current: {mean_reward:.4f})")
            
            if no_improve_count >= PATIENCE:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING after {PATIENCE} iterations without improvement")
                print(f"Best reward: {best_reward:.4f}")
                print(f"{'='*60}\n")
                ppo_runner.save(os.path.join(ppo_runner.log_dir, f'model_early_stop_{iteration}.pt'))
                sys.exit(0)
        
        ppo_runner.log = log_with_early_stop
        print(f"\n[Early Stopping] Patience: {PATIENCE}, Min delta: {MIN_DELTA}\n")
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
