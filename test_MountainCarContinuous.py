import gym
from tqdm import tqdm
from collections import deque
from reinforce import Reinforce
from ddpg import OUNoise, DDPGAgent
import optparse
import numpy as np
import torch

# Ensure compatibility with older numpy versions
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool

def test_ddpg(agent, env, noise, max_steps, num_episodes=10):
    """
    Test the DDPG algorithm on the environment.

    Args:
        agent (DDPGAgent): Pretrained DDPG agent.
        env (gym.Env): The environment to test on.
        noise (OUNoise): Noise generator for exploration.
        max_steps (int): Maximum steps per episode.
        num_episodes (int): Number of episodes to test.
    """
    # IMPLEMENTATION REMOVED: DDPG testing loop
    rewards = []
    print(f"Average Reward over {num_episodes} episodes: {np.mean(rewards):.2f}")


def test_reinforce(network, env, max_steps, num_episodes=10):
    """
    Test the REINFORCE algorithm on the environment.

    Args:
        network (Reinforce): Pretrained REINFORCE network.
        env (gym.Env): The environment to test on.
        max_steps (int): Maximum steps per episode.
        num_episodes (int): Number of episodes to test.
    """
    # IMPLEMENTATION REMOVED: REINFORCE testing loop
    rewards = []
    print(f"Average Reward over {num_episodes} episodes: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = optparse.OptionParser("Test MountainCarContinuous using Reinforce or DDPG.")
    parser.add_option("-a", dest="agent", choices=["reinforce", "ddpg"], help="Algorithm to test")
    options, args = parser.parse_args()

    # Environment setup
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    MAX_STEPS = 500

    if options.agent == "ddpg":
        # Load DDPG agent
        agent = DDPGAgent(env)
        agent.test_actor("mountaincar_ddpg_")
        noise_generator = OUNoise(env.action_space)

        # Test DDPG
        test_ddpg(agent, env, noise_generator, MAX_STEPS)

    elif options.agent == "reinforce":
        # Load REINFORCE agent
        policy_network = Reinforce(env.observation_space.shape[0])
        policy_network.load_state_dict(torch.load('mountaincar_reinforce.pt', weights_only=True))

        # Test REINFORCE
        test_reinforce(policy_network, env, MAX_STEPS)

    env.close()
