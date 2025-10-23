"""
Test the site gym environment.

TODO: move to testing dir for more rigorous tests
"""
import gymnasium as gym
from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentSiteEnv
from web_agent_site.models import (
    HumanPolicy,
    RandomPolicy,
)
from web_agent_site.utils import DEBUG_PROD_SIZE


if __name__ == '__main__':
    #env = gym.make('WebAgentSite-v0')
    #env = WebAgentSiteEnv(render=True, pause=2.0)
    #env = WebAgentSiteEnv(observation_mode='html', render=False)
    try:
        env = WebAgentSiteEnv(observation_mode='text', render=False, num_products=DEBUG_PROD_SIZE)
    except FileNotFoundError as e:
        if 'chromedriver' in str(e).lower():
            print("\n❌ ChromeDriver not found!")
            print("\nTo run the site environment, you need to:")
            print("  1. Download ChromeDriver for your system from:")
            print("     https://chromedriver.chromium.org/downloads")
            print("  2. Or use: sudo apt-get install chromium-chromedriver (on Ubuntu/Debian)")
            print("  3. Place the binary at: web_agent_site/envs/chromedriver")
            print("     Or ensure 'chromedriver' is in your PATH")
            exit(1)
        raise
    except OSError as e:
        if 'Exec format error' in str(e):
            print("\n❌ ChromeDriver architecture mismatch!")
            print("\nThe chromedriver binary is not compatible with your system.")
            print("Please download the correct version for your OS and architecture:")
            print("  - Linux: https://chromedriver.chromium.org/downloads")
            print("  - Or use: sudo apt-get install chromium-chromedriver (on Ubuntu/Debian)")
            exit(1)
        raise
    except Exception as e:
        if 'ERR_CONNECTION_REFUSED' in str(e) or 'Connection refused' in str(e):
            print("\n❌ Cannot connect to WebShop server!")
            print("\nThe site environment requires the Flask app to be running.")
            print("\nTo fix this:")
            print("  1. In one terminal, start the Flask app:")
            print("     make run-dev")
            print("\n  2. In another terminal, run this script again:")
            print("     make run-web-agent-site")
            print("     # or: ./run_web_agent_site_env.sh")
            exit(1)
        raise
    
    global_step = 0
    observation, info = env.reset()
    
    try:
        #policy = HumanPolicy()
        policy = RandomPolicy()
    
        while True:
            print(observation)
            available_actions = env.get_available_actions()
            print('Available actions:', available_actions)
            action = policy.forward(observation, available_actions)
            # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            if done:
                break
            global_step += 1
    finally:
        env.close()