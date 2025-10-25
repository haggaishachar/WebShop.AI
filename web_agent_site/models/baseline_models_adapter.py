"""
Adapter utilities for baseline IL/RL models.

This module provides shared utilities extracted from the baseline_models/
directory to support the IL and RL policies. It adapts the original code
to work with the WebShop policy interface.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# Add baseline_models to path for imports
BASELINE_MODELS_DIR = Path(__file__).parent.parent.parent / "baseline_models"
if str(BASELINE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_MODELS_DIR))


def process(s: str) -> str:
    """
    Process state/action strings for BERT tokenization.
    
    Adapted from baseline_models/train_choice_il.py
    Normalizes strings by lowercasing and removing quotes.
    """
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s


def process_goal(state: str) -> str:
    """
    Extract and process the goal/instruction from the state.
    
    Adapted from baseline_models/train_choice_il.py
    Removes UI elements and extracts just the instruction text.
    """
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    
    # Remove price constraint for search query generation
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    
    return state


class BaselineTokenizer:
    """
    Wrapper for BERT tokenizer with special tokens for WebShop.
    
    Adds custom tokens for button states used in the baseline models.
    """
    
    def __init__(self):
        """Initialize tokenizer with special WebShop tokens."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            truncation_side='left'
        )
        
        # Add special tokens used in WebShop
        special_tokens = [
            '[button]',
            '[button_]',
            '[clicked button]',
            '[clicked button_]'
        ]
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        
        # Store vocab size after adding tokens
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text: Union[str, List[str]], max_length: int = 512, **kwargs):
        """
        Encode text to token IDs.
        
        Args:
            text: String or list of strings to encode
            max_length: Maximum sequence length
            **kwargs: Additional arguments for tokenizer
            
        Returns:
            Encoding dict with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            **kwargs
        )
    
    def encode_single(self, observation: str, max_length: int = 512) -> List[int]:
        """
        Encode a single observation to token IDs list.
        
        Args:
            observation: Text to encode
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        observation = observation.lower().replace('"', '').replace("'", "").strip()
        observation = observation.replace('[sep]', '[SEP]')
        token_ids = self.tokenizer.encode(
            observation,
            truncation=True,
            max_length=max_length
        )
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded string
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        # Clean up spacing around brackets
        text = text.replace(' [ ', '[').replace(' ]', ']')
        return text


def collate_batch_for_bert(
    state_text: str,
    valid_actions: List[str],
    image_feat: Optional[torch.Tensor] = None,
    tokenizer: Optional[BaselineTokenizer] = None
) -> dict:
    """
    Create a batch dictionary for BERT choice model inference.
    
    Args:
        state_text: Current state observation
        valid_actions: List of valid action strings
        image_feat: Optional image features tensor
        tokenizer: Tokenizer to use (creates new if None)
        
    Returns:
        Dictionary with tensors ready for model input
    """
    if tokenizer is None:
        tokenizer = BaselineTokenizer()
    
    # Process state and actions
    state_processed = process(state_text)
    actions_processed = [process(act) for act in valid_actions]
    
    # Encode state and actions
    state_encodings = tokenizer.encode(state_processed, max_length=512)
    action_encodings = tokenizer.encode(actions_processed, max_length=512)
    
    batch = {
        'state_input_ids': torch.tensor([state_encodings['input_ids']]),
        'state_attention_mask': torch.tensor([state_encodings['attention_mask']]),
        'action_input_ids': torch.tensor(action_encodings['input_ids']),
        'action_attention_mask': torch.tensor(action_encodings['attention_mask']),
        'sizes': torch.tensor([len(valid_actions)]),
    }
    
    # Add image features if provided
    if image_feat is not None:
        if isinstance(image_feat, list):
            image_feat = torch.tensor(image_feat)
        batch['images'] = image_feat.unsqueeze(0) if image_feat.dim() == 1 else image_feat
    
    return batch


def select_action_from_logits(
    logits: torch.Tensor,
    valid_actions: List[str],
    method: str = 'greedy'
) -> tuple[str, int]:
    """
    Select an action from model logits.
    
    Args:
        logits: Model output logits for actions
        valid_actions: List of valid action strings
        method: Selection method ('greedy', 'softmax', or 'epsilon')
        
    Returns:
        Tuple of (selected_action_string, selected_index)
    """
    if method == 'softmax':
        # Stochastic sampling
        probs = F.softmax(logits, dim=0)
        idx = torch.multinomial(probs, num_samples=1)[0].item()
    elif method == 'greedy':
        # Greedy selection
        idx = logits.argmax(0).item()
    else:
        # Default to greedy
        idx = logits.argmax(0).item()
    
    return valid_actions[idx], idx


def check_baseline_models_installed() -> tuple[bool, str]:
    """
    Check if baseline models are downloaded and available.
    
    Returns:
        Tuple of (models_available, message)
    """
    choice_model_path = BASELINE_MODELS_DIR / "ckpts" / "web_click" / "epoch_9" / "model.pth"
    search_model_path = BASELINE_MODELS_DIR / "ckpts" / "web_search" / "checkpoint-800"
    
    if not choice_model_path.exists() or not search_model_path.exists():
        message = (
            "Baseline models not found!\n\n"
            "Please download them by running:\n"
            "  make download-baseline-models\n\n"
            "Or manually:\n"
            "  python baseline_models/download_models.py"
        )
        return False, message
    
    return True, "Models found"


def get_model_paths() -> dict:
    """
    Get paths to baseline model checkpoints.
    
    Returns:
        Dictionary with 'choice_model' and 'search_model' paths
    """
    return {
        'choice_model': str(BASELINE_MODELS_DIR / "ckpts" / "web_click" / "epoch_9" / "model.pth"),
        'search_model': str(BASELINE_MODELS_DIR / "ckpts" / "web_search" / "checkpoint-800"),
    }


