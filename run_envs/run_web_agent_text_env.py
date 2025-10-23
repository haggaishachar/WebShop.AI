"""
Test the text gym environment.

TODO: move to testing dir for more rigorous tests
"""
import gymnasium as gym
from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

if __name__ == '__main__':
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)
    observation, info = env.reset()
    
    try:
        policy = RandomPolicy()
    
        while True:
            print(observation)
            available_actions = env.unwrapped.get_available_actions()
            print('Available actions:', available_actions)
            action = policy.forward(observation, available_actions)
            # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            if done:
                break
    finally:
        env.close()