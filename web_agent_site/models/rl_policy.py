"""
Reinforcement Learning (RL) Policy for WebShop.

This policy wraps the Agent class from baseline_models/agent.py,
which supports both RNN and BERT-based RL models trained with
policy gradient methods.
"""

import sys
from pathlib import Path
from typing import Optional, Literal
from collections import namedtuple

import torch

from .models import BasePolicy
from .baseline_models_adapter import (
    check_baseline_models_installed,
    get_model_paths,
    BASELINE_MODELS_DIR,
)

# Add baseline_models to path
if str(BASELINE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_MODELS_DIR))

try:
    from agent import Agent
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error importing baseline models: {e}")
    raise

# State namedtuple from baseline_models/agent.py
State = namedtuple('State', ('obs', 'goal', 'click', 'estimate', 'obs_str', 'goal_str', 'image_feat'))


class SimpleArgs:
    """Simple argument container for Agent initialization."""
    
    def __init__(
        self,
        network: str = 'bert',
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        arch_encoder: str = 'bert',
        grad_encoder: bool = False,
        gru_embed: int = 128,
        get_image: bool = False,
        bert_path: str = '',
        output_dir: str = './ckpts',
        clip: float = 5.0,
        w_pg: float = 1.0,
        w_td: float = 1.0,
        w_il: float = 0.0,
        w_en: float = 0.0,
        learning_rate: float = 1e-4,
        gamma: float = 0.9,
    ):
        """Initialize with default arguments for Agent."""
        self.network = network
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.arch_encoder = arch_encoder
        self.grad_encoder = grad_encoder
        self.gru_embed = gru_embed
        self.get_image = get_image
        self.bert_path = bert_path
        self.output_dir = output_dir
        self.clip = clip
        self.w_pg = w_pg
        self.w_td = w_td
        self.w_il = w_il
        self.w_en = w_en
        self.learning_rate = learning_rate
        self.gamma = gamma


class RLPolicy(BasePolicy):
    """
    Reinforcement Learning Policy using baseline Agent.
    
    This policy wraps the Agent class from baseline_models, which supports
    policy gradient training and both RNN and BERT architectures.
    
    Args:
        model_path: Path to trained RL model checkpoint (optional)
        network: Network type - 'bert' or 'rnn' (default: 'bert')
        action_method: Action selection - 'softmax', 'greedy', or 'eps' (default: 'greedy')
        use_images: Whether to use image features (default: False)
        device: Device to run on ('cuda' or 'cpu', default: auto-detect)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        network: Literal['bert', 'rnn'] = 'bert',
        action_method: Literal['softmax', 'greedy', 'eps'] = 'greedy',
        use_images: bool = False,
        device: Optional[str] = None,
        eps: float = 0.1,
    ):
        """Initialize RL policy with Agent."""
        super().__init__()
        
        # Configuration
        self.action_method = action_method
        self.use_images = use_images
        self.eps = eps
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initializing RL Policy on {self.device}...")
        
        # Create args for Agent
        args = SimpleArgs(
            network=network,
            get_image=use_images,
            bert_path='',  # Will load from checkpoint if provided
        )
        
        # Initialize Agent
        self.agent = Agent(args)
        
        # Move agent network to correct device
        self.agent.network = self.agent.network.to(self.device)
        
        # Patch the rl_forward method to use the correct device instead of hardcoded cuda()
        self._patch_device_handling()
        
        # Load checkpoint if provided
        if model_path:
            print(f"Loading RL model from {model_path}...")
            self.agent.network.load_state_dict(
                torch.load(model_path, map_location=self.device),
                strict=False
            )
            print("✓ RL model loaded")
        else:
            # Try to load IL model as starting point
            models_available, _ = check_baseline_models_installed()
            if models_available:
                model_paths = get_model_paths()
                try:
                    print("Loading IL choice model as initialization...")
                    self.agent.network.load_state_dict(
                        torch.load(model_paths['choice_model'], map_location=self.device),
                        strict=False
                    )
                    print("✓ IL choice model loaded as initialization")
                except Exception as e:
                    print(f"⚠️  Could not load IL model: {e}")
                    print("Using randomly initialized network")
        
        self.agent.network.eval()
        
        # State tracking
        self.current_goal = None
        self.prev_observation = None
        
        print(f"✓ RL Policy initialized (network={network}, method={action_method}, images={use_images})")
    
    def _patch_device_handling(self):
        """
        Patch the rl_forward method to use self.device instead of hardcoded .cuda() calls.
        This is necessary because the baseline models have hardcoded cuda() calls.
        """
        original_rl_forward = self.agent.network.rl_forward
        device = self.device
        
        def patched_rl_forward(state_batch, act_batch, value=False, q=False, act=False):
            """Patched version that uses the correct device."""
            import torch.nn as nn
            import torch.nn.functional as F
            
            act_values = []
            act_sizes = []
            values = []
            for state, valid_acts in zip(state_batch, act_batch):
                with torch.set_grad_enabled(not act):
                    state_ids = torch.tensor([state.obs]).to(device)
                    state_mask = (state_ids > 0).int()
                    act_lens = [len(_) for _ in valid_acts]
                    act_ids = [torch.tensor(_) for _ in valid_acts]
                    act_ids = nn.utils.rnn.pad_sequence(act_ids, batch_first=True).to(device)
                    act_mask = (act_ids > 0).int()
                    act_size = torch.tensor([len(valid_acts)]).to(device)
                    
                    # Handle images
                    if self.agent.network.image_linear is not None:
                        images = [state.image_feat]
                        images = [torch.zeros(512) if _ is None else _ for _ in images]
                        images = torch.stack(images).to(device)
                    else:
                        images = None
                    
                    logits = self.agent.network.forward(
                        state_ids, state_mask, act_ids, act_mask, act_size, images=images
                    ).logits[0]
                    act_values.append(logits)
                    act_sizes.append(len(valid_acts))
                
                if value:
                    v = self.agent.network.bert(state_ids, state_mask)[0]
                    values.append(self.agent.network.linear_3(v[0][0]))
            
            act_values = torch.cat(act_values, dim=0)
            act_values = torch.cat([F.log_softmax(_, dim=0) for _ in act_values.split(act_sizes)], dim=0)
            
            if value:
                values = torch.cat(values, dim=0)
                return act_values, act_sizes, values
            else:
                return act_values, act_sizes
        
        # Replace the method
        self.agent.network.rl_forward = patched_rl_forward
    
    def reset(self):
        """Reset policy state for a new episode."""
        self.current_goal = None
        self.prev_observation = None
    
    def forward(self, observation: str, available_actions: dict) -> str:
        """
        Generate action using RL agent.
        
        Args:
            observation: Current observation text
            available_actions: Dict with 'has_search_bar' and 'clickables'
            
        Returns:
            Action string
        """
        # Extract goal from observation if this is the first step
        if self.current_goal is None:
            self.current_goal = self._extract_goal(observation)
        
        # Build state for agent
        state = self._build_state(observation, available_actions)
        
        # Get valid actions as strings
        if available_actions.get('has_search_bar'):
            # For search, we'll use the goal as the query
            # In full RL, you'd use a separate search model
            # Simply return a search action with the goal
            action = f'search[{self.current_goal}]'
        else:
            clickables = available_actions.get('clickables', [])
            if not clickables:
                return 'search[product]'
            valid_acts = [[f'click[{c}]' for c in clickables]]
            
            # Get action from agent
            with torch.no_grad():
                act_strs, act_ids, values = self.agent.act(
                    [state],
                    valid_acts,
                    method=self.action_method,
                    eps=self.eps
                )
            
            action = act_strs[0]
        
        # Update state
        self.prev_observation = observation
        
        return action
    
    def _extract_goal(self, observation: str) -> str:
        """
        Extract goal/instruction from observation.
        
        Args:
            observation: Full observation text
            
        Returns:
            Goal string
        """
        # Look for instruction in observation
        if 'Instruction:' in observation:
            parts = observation.split('Instruction:')
            if len(parts) > 1:
                # Extract text between Instruction: and Search
                goal_part = parts[1].split('Search')[0].strip()
                if not goal_part:
                    # Try alternative split by newlines
                    lines = [l.strip() for l in parts[1].strip().split('\n') if l.strip()]
                    if lines:
                        goal_part = ' '.join(lines[:3])  # Take first few lines
                
                # Clean up the goal
                if goal_part:
                    # Remove [SEP] tokens and extra whitespace
                    goal_part = goal_part.replace('[SEP]', '').replace('[sep]', '')
                    goal_part = ' '.join(goal_part.split())  # Normalize whitespace
                    # Keep it simple - just extract key product terms
                    # Remove "find me" and convert to search-friendly format
                    goal_part = goal_part.replace('find me', '').strip()
                    return goal_part[:200]  # Reasonable limit
        
        # Fallback - look for the first meaningful line after WebShop
        lines = [line.strip() for line in observation.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            if 'Instruction:' in line and i + 1 < len(lines):
                goal_line = lines[i + 1]
                if goal_line and goal_line.lower() not in ['search', 'webshop']:
                    goal_line = goal_line.replace('[SEP]', '').replace('[sep]', '').strip()
                    goal_line = ' '.join(goal_line.split())
                    return goal_line[:200]
        
        # Last resort fallback
        return "product"
    
    def _build_state(self, observation: str, available_actions: dict) -> State:
        """
        Build State namedtuple for agent.
        
        Args:
            observation: Current observation
            available_actions: Available actions dict
            
        Returns:
            State namedtuple
        """
        # Encode observation
        obs_ids = self.agent.encode(observation)
        goal_ids = self.agent.encode(self.current_goal)
        
        # Check if we're on an item page (has clickable actions)
        click = not available_actions.get('has_search_bar', False)
        
        # Estimate score (placeholder, would need env access for real score)
        estimate = 0.0
        
        # String representations
        obs_str = observation.replace('\n', '[SEP]')
        goal_str = self.current_goal
        
        # Image features (placeholder if not using images)
        image_feat = None
        if self.use_images:
            # In a full implementation, this would come from the environment
            image_feat = torch.zeros(512)
        
        return State(
            obs=obs_ids,
            goal=goal_ids,
            click=click,
            estimate=estimate,
            obs_str=obs_str,
            goal_str=goal_str,
            image_feat=image_feat
        )


def create_rl_policy(**kwargs) -> RLPolicy:
    """
    Factory function to create RL policy with error handling.
    
    Args:
        **kwargs: Arguments to pass to RLPolicy constructor
        
    Returns:
        RLPolicy instance
        
    Raises:
        FileNotFoundError: If models are not found
        ImportError: If required packages are missing
    """
    try:
        return RLPolicy(**kwargs)
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        raise
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nPlease install baseline dependencies:")
        print("  uv pip install -e .[baseline]")
        print()
        raise

