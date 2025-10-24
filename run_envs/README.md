# WebShop Environment Runners

This directory contains scripts for running WebShop environments with different configurations.

## Available Runners

### 1. `run_web_agent_env.py` - Configurable Runner (New)

A flexible runner that accepts configurable parameters for observation mode, number of products, and policy.

#### Command Line Usage

```bash
# Run with default parameters (text mode, DEBUG_PROD_SIZE products, random policy)
python run_envs/run_web_agent_env.py

# Run with custom parameters
python run_envs/run_web_agent_env.py --observation-mode text --num-products 100 --policy random

# Run with HTML observation mode
python run_envs/run_web_agent_env.py --observation-mode html --num-products 1000 --policy human

# Use the site environment (requires ChromeDriver and running Flask server)
python run_envs/run_web_agent_env.py --use-site-env --observation-mode text --policy random
```

#### Parameters

- `--observation-mode`: Observation mode (default: `text`)
  - Options: `text`, `html`, `text_rich`
  
- `--num-products`: Number of products to use (default: `DEBUG_PROD_SIZE` from utils.py, which is `None`)
  - Set to an integer to limit the number of products
  - Common values: `100`, `1000`, `100000`
  
- `--policy`: Policy to use (default: `random`)
  - Options: `random`, `human`
  
- `--num-episodes`: Number of episodes to run (default: `1`)
  - Set to run multiple episodes and get aggregate statistics
  - Example: `--num-episodes 5`
  
- `--use-site-env`: Use WebAgentSiteEnv instead of WebAgentTextEnv (default: False)
  - Requires ChromeDriver and a running Flask server

#### Python API Usage

You can also use the runner programmatically:

```python
from run_envs.run_web_agent_env import create_env, run_episode
from web_agent_site.models import RandomPolicy

# Create environment with custom parameters
env = create_env(
    observation_mode='text',
    num_products=100,
    use_site_env=False  # Use text environment
)

# Create policy
policy = RandomPolicy()

# Run an episode
stats = run_episode(env, policy)
print(f"Total reward: {stats['total_reward']}")
print(f"Total steps: {stats['steps']}")

# Clean up
env.close()
```

### 2. `run_web_agent_text_env.py` - Text Environment Runner

Original runner for the text-based environment with hardcoded parameters.

```bash
python run_envs/run_web_agent_text_env.py
```

### 3. `run_web_agent_site_env.py` - Site Environment Runner

Original runner for the site-based environment (requires ChromeDriver and Flask server).

```bash
python run_envs/run_web_agent_site_env.py
```

## Environment Types

### TextEnv (Default)
- Lightweight simulated environment
- No browser required
- Faster execution
- Text or HTML observations

### SiteEnv
- Uses real browser (ChromeDriver)
- Requires Flask server running
- More realistic but slower
- Useful for testing visual aspects

## Requirements

### For TextEnv
- Python packages (installed via pyproject.toml)
- No additional setup required

### For SiteEnv
- ChromeDriver installed and in PATH
- Flask server running (`make run-dev`)
- See main README for ChromeDriver installation instructions

## Makefile Shortcuts

The makefile provides convenient shortcuts for common use cases:

```bash
# Run text environment with random policy (default)
make run-web-agent-text

# Run site environment with random policy (requires ChromeDriver and Flask server)
make run-web-agent-site

# Run text environment with human policy (interactive)
make run-web-agent-human

# Run with custom parameters
make run-web-agent-custom ARGS='--observation-mode text --num-products 100 --policy random'

# Run multiple episodes
make run-web-agent-custom ARGS='--num-episodes 5'
```

## Examples

### Quick test with text environment
```bash
python run_envs/run_web_agent_env.py
# or
make run-web-agent-text
```

### Test with specific number of products
```bash
python run_envs/run_web_agent_env.py --num-products 100
# or
make run-web-agent-custom ARGS='--num-products 100'
```

### Interactive human policy
```bash
python run_envs/run_web_agent_env.py --policy human
# or
make run-web-agent-human
```

### Run multiple episodes for evaluation
```bash
# Run 5 episodes and get aggregate statistics
python run_envs/run_web_agent_env.py --num-episodes 5

# Run 10 episodes with 100 products
python run_envs/run_web_agent_env.py --num-episodes 10 --num-products 100
```

### Full test with site environment
```bash
# Terminal 1: Start Flask server
make run-dev

# Terminal 2: Run agent
python run_envs/run_web_agent_env.py --use-site-env --policy random
# or
make run-web-agent-site
```

