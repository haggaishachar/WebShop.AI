# 🛒 WebShop

[![Python version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Princeton-orange)](https://copyright.princeton.edu/policy)
[![PyPI version](https://badge.fury.io/py/webshop.svg)](https://badge.fury.io/py/webshop)
![Pytest workflow](https://github.com/princeton-nlp/webshop/actions/workflows/pytest.yml/badge.svg)

Implementation of the WebShop environment and search agents for the paper:

**[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://webshop-pnlp.github.io/)**  
[Shunyu Yao*](https://ysymyth.github.io/), [Howard Chen*](https://howard50b.github.io/), [John Yang](https://john-b-yang.github.io/), [Karthik Narasimhan](https://www.cs.princeton.edu/~karthikn/)

<p float="left">
  <img src="assets/diagram.gif">
</p>

This repository contains code for reproducing results. If you find this work useful in your research, please cite:

```
@inproceedings{yao2022webshop,
  bibtex_show = {true},
  title = {WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents},
  author = {Yao, Shunyu and Chen, Howard and Yang, John and Narasimhan, Karthik},
  booktitle = {ArXiv},
  year = {preprint},
  html = {https://arxiv.org/abs/2207.01206},
  tag = {NLP}
}
```
## 📖 Table of Contents <!-- omit in toc -->
* [👋 Overview](#-overview)
* [🚀 Setup](#-setup)
* [🛠️ Usage](#-usage)
* [💫 Contributions](#-contributions)
* [🪪 License](#-license)
## 👋 Overview
WebShop is a simulated e-commerce website environment with 1.18 million real-world products and 12,087 crowd-sourced text instructions. In this environment, an agent needs to navigate multiple types of webpages and issue diverse actions to find, customize, and purchase a product given an instruction. WebShop provides several challenges including understanding compositional instructions, query (re-)formulation, dealing with noisy text in webpages, and performing strategic exploration.

**Hugging Face Demo**: Devise your own natural language query for a product and ask for an agent trained with WebShop to find it on Amazon or eBay, deployed as a 🤗 Hugging Face space [here](https://huggingface.co/spaces/webshop/amazon_shop)!

## 🚀 Setup
This project uses modern Python tooling with [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Prerequisites
1. **Python 3.12+**: [Download here](https://www.python.org/downloads/)
2. **Java**: [Download here](https://www.java.com/en/download/) (required for search engine)
3. **(Optional) uv package manager**: Will be auto-installed by `make setup`, or [install manually](https://github.com/astral-sh/uv#installation)

### Quick Start
1. Clone the repository:
```sh
git clone https://github.com/princeton-nlp/webshop.git webshop
cd webshop
```

2. Run the automated setup:
```sh
make setup
```
This command will:
* Install [uv](https://github.com/astral-sh/uv) if not already installed (10-100x faster than pip)
* Create a virtual environment (`.venv`) and install all dependencies from `pyproject.toml`
* Download and install the spaCy `en_core_web_sm` language model

3. Download data and build search indexes:
```sh
# Option A: Download small dataset (1,000 products) - faster for testing
make setup-data-small

# Option B: Download full dataset (all products)
make setup-data-all

# Build search engine indexes (required after downloading data)
make setup-search-engine
```

4. (Optional) Download human trajectories:
```sh
make setup-human-trajs
```

### Available Make Targets
The project uses a `Makefile` for common tasks:

**Setup Commands:**
- `make setup` - Complete initial setup (uv, dependencies, spacy model)
- `make setup-data-small` - Download 1,000 product dataset
- `make setup-data-all` - Download full product dataset
- `make setup-search-engine` - Build search engine indexes
- `make setup-human-trajs` - Download human demonstration trajectories
- `make download-spacy-model-lg` - Download larger spaCy model (en_core_web_lg)

**Run Commands:**
- `make run-dev` - Start Flask webapp in development mode
- `make run-prod` - Start Flask webapp in production mode
- `make run-web-agent-site` - Run web agent with browser (requires ChromeDriver)
- `make run-web-agent-text` - Run web agent text environment

**Utility Commands:**
- `make clean` - Clean up temporary files
- `make check-uv` - Verify uv is installed
- `make check-search-engine` - Verify search engine indexes are built

### Modern Python Tooling
This project has been modernized to use:
- **[uv](https://github.com/astral-sh/uv)**: Ultra-fast Python package manager (10-100x faster than pip)
- **pyproject.toml**: Modern Python project configuration (replaces requirements.txt)
- **Gymnasium**: Updated OpenAI Gym API with proper typing and modern standards
- **Python 3.12+**: Latest Python features and performance improvements

All dependencies are managed in `pyproject.toml`. To manually sync dependencies:
```sh
uv sync
```

To run Python commands in the virtual environment:
```sh
uv run python your_script.py
```

### Loading All Products
By default, the WebShop only loads 1,000 products for faster environment preview. To load all products, change `web_agent_site/utils.py`:
```python
# DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2_1000.json')
# DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle_1000.json')
DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')
```

### Optional Downloads
- **Image Features**: Download ResNet image feature files [here](https://drive.google.com/drive/folders/1jglJDqNV2ryrlZzrS0yOEk-aRAcLAhNw?usp=sharing) and put into `data/` for running models that require image features.
- **Human Demonstrations**: Download human demonstration data [here](https://drive.google.com/file/d/1GWC8UlUzfT9PRTRxgYOwuKSJp4hyV1dp/view?usp=sharing).

## 🛠️ Usage
The WebShop environment can be rendered in two modes - `html` and `simple` - each of which offer a different observation space. The `simple` mode strips away the extraneous meta-data that the `html` mode includes to make model training and evaluation easier.

### Webpage Environment (`html` mode)
Launch the `WebShop` webpage:
```sh
make run-dev
```
The site will be viewable in your browser at http://localhost:3000/. Navigate to http://localhost:3000/ABC to land on the search home page with a random instruction.

Navigating the website will automatically generate a corresponding trajectory file in the `user_session_logs/mturk` folder. Each file corresponds to a single instruction/web session, and each step of the file corresponds to a single action (i.e. `search[...]`, `click[...]`).

The `run-dev` command includes:
* `--log`: Creates trajectory `.jsonl` log files of actions on WebShop
* `--attrs`: Displays an `Attributes` tab on the `item_page` of WebShop

For production mode (logging only, no attributes):
```sh
make run-prod
```

### Text Environment (`simple` mode)
The `simple` mode of the WebShop environment is packaged and readily available as a Gymnasium environment. The environment definitions can be found in the `web_agent_site/envs` folder.

To start using the environment and building agents that interact with WebShop, include the following statements in your Python file:
```python
import gymnasium as gym
from web_agent_site.envs import WebAgentTextEnv

env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=...)
observation, info = env.reset()

# Your agent logic here
action = your_policy(observation)
observation, reward, terminated, truncated, info = env.step(action)
```
Now, you can write your own agent that interacts with the environment via the standard Gymnasium [interface](https://gymnasium.farama.org/).

**Note**: This project uses [Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym). The API follows the Gymnasium v1.0+ standard with 5-value returns from `step()`: `(observation, reward, terminated, truncated, info)`.

### Running Example Agents
Examples of a `RandomPolicy` agent interacting with the WebShop environment in both `html` and `simple` modes can be found in the `run_envs` folder. To run these examples locally:

**Text Environment:**
```sh
make run-web-agent-text
# or
uv run python run_envs/run_web_agent_text_env.py
```

**Site Environment (with browser):**
```sh
make run-web-agent-site
# or
uv run python run_envs/run_web_agent_site_env.py
```

Output example:
```
Products loaded.
Keys Cleaned.
Attributes Loaded.
100%|██████████████████| 1000/1000
Loaded 6910 goals.
Amazon Shopping Game [SEP] Instruction: [SEP] Find me slim f...
Available actions: {'has_search_bar': True, 'clickables': ['search']}
Taking action "search[shoes]" -> Reward = 0.0
...
```

**ChromeDriver Setup:**
The site environment requires ChromeDriver. See `web_agent_site/envs/README.md` for installation instructions:
- **Recommended**: Install system-wide (`sudo apt-get install chromium-chromedriver` on Ubuntu/Debian)
- **Alternative**: Download from [ChromeDriver](https://chromedriver.chromium.org/downloads) and place in `web_agent_site/envs/chromedriver`

### Baseline Models
To run baseline models (rule, IL, RL, IL+RL) from the paper, please refer to the `README.md` in the [baseline_models](https://github.com/princeton-nlp/webshop/tree/master/baseline_models) folder.

### Sim-to-real Transfer
To read more about how the sim-to-real transfer of agents trained on WebShop to other environments works, please refer to the `README.md` in the [transfer](https://github.com/princeton-nlp/webshop/tree/master/transfer) folder.

## 💫 Contributions
We would love to hear from the broader NLP and Machine Learning community, and we welcome any contributions, pull requests, or issues! To do so, please either file a new pull request or issue and fill in the corresponding templates accordingly. We'll be sure to follow up shortly!

## 🪪 License
Check `LICENSE.md`
