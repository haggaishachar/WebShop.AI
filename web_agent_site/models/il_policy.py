"""
Imitation Learning (IL) Policy for WebShop.

This policy combines two pre-trained models:
1. BART model for search query generation
2. BERT model for action selection

Based on the baseline models from the WebShop paper.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .models import BasePolicy
from .baseline_models_adapter import (
    BaselineTokenizer,
    process,
    process_goal,
    collate_batch_for_bert,
    select_action_from_logits,
    check_baseline_models_installed,
    get_model_paths,
    BASELINE_MODELS_DIR,
)

# Add baseline_models to path
if str(BASELINE_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_MODELS_DIR))

try:
    from transformers import BartForConditionalGeneration, BartTokenizer
    from models.bert import BertModelForWebshop, BertConfigForWebshop
except ImportError as e:
    print(f"Error importing baseline models: {e}")
    print("Make sure transformers is installed and baseline_models is accessible.")
    raise


class ILPolicy(BasePolicy):
    """
    Imitation Learning Policy using pre-trained BART and BERT models.
    
    This policy implements a two-stage approach:
    1. Search stage: BART generates search queries from instructions
    2. Choice stage: BERT selects actions from available options
    
    Args:
        use_search_model: If True, use BART for search query generation.
                         If False, use the instruction directly (default: True)
        use_images: If True, incorporate image features (default: False)
        memory: If True, track previous observations and actions (default: False)
        sampling_method: Action selection method - 'greedy' or 'softmax' (default: 'greedy')
        device: Device to run models on ('cuda' or 'cpu', default: auto-detect)
    """
    
    def __init__(
        self,
        use_search_model: bool = True,
        use_images: bool = False,
        memory: bool = False,
        sampling_method: str = 'greedy',
        device: Optional[str] = None
    ):
        """Initialize the IL policy with pre-trained models."""
        super().__init__()
        
        # Check if models are available
        models_available, message = check_baseline_models_installed()
        if not models_available:
            raise FileNotFoundError(message)
        
        # Configuration
        self.use_search_model = use_search_model
        self.use_images = use_images
        self.memory = memory
        self.sampling_method = sampling_method
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initializing IL Policy on {self.device}...")
        
        # Get model paths
        model_paths = get_model_paths()
        
        # Initialize tokenizers
        self.tokenizer = BaselineTokenizer()
        
        # Load BERT choice model
        print("Loading BERT choice model...")
        config = BertConfigForWebshop(image=self.use_images, pretrained_bert=True)
        self.choice_model = BertModelForWebshop(config)
        self.choice_model.load_state_dict(
            torch.load(model_paths['choice_model'], map_location=self.device),
            strict=False
        )
        self.choice_model.to(self.device)
        self.choice_model.eval()
        print("✓ BERT choice model loaded")
        
        # Load BART search model (optional)
        self.search_model = None
        self.bart_tokenizer = None
        if self.use_search_model:
            print("Loading BART search model...")
            self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            self.search_model = BartForConditionalGeneration.from_pretrained(
                model_paths['search_model']
            )
            self.search_model.to(self.device)
            self.search_model.eval()
            print("✓ BART search model loaded")
        
        # Memory tracking
        self.prev_observations = []
        self.prev_actions = []
        
        print(f"✓ IL Policy initialized (search={use_search_model}, images={use_images}, memory={memory})")
    
    def reset(self):
        """Reset policy state for a new episode."""
        self.prev_observations = []
        self.prev_actions = []
    
    def forward(self, observation: str, available_actions: dict) -> str:
        """
        Generate an action based on observation and available actions.
        
        Args:
            observation: Current observation text
            available_actions: Dict with 'has_search_bar' and 'clickables'
            
        Returns:
            Action string (e.g., 'search[query]' or 'click[item]')
        """
        # Encode state with memory if enabled
        state = self._encode_state(observation)
        
        # Check if we're at search stage
        if available_actions.get('has_search_bar'):
            action = self._predict_search(observation)
        else:
            # Get valid actions
            clickables = available_actions.get('clickables', [])
            if not clickables:
                return 'search[product]'  # Fallback
            
            # Format actions as click[option] for the choice model
            valid_acts = [f'click[{c}]' for c in clickables]
            
            # Predict choice
            action = self._predict_choice(state, valid_acts)
        
        # Update memory if enabled
        if self.memory:
            self.prev_observations.append(observation)
            self.prev_actions.append(action)
        
        return action
    
    def _encode_state(self, observation: str) -> str:
        """
        Encode state with memory if enabled.
        
        Args:
            observation: Current observation
            
        Returns:
            Encoded state string
        """
        if not self.memory or not self.prev_observations:
            return observation
        
        # Construct state with previous observations and actions
        # Following the baseline_models/env.py pattern
        text_list = [observation]
        
        # Add previous actions and observations
        max_history = max(1, len(self.prev_actions))  # At least 1
        for i in range(1, 1 + max_history):
            if len(self.prev_actions) >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_observations) >= i:
                text_list.append(self.prev_observations[-i])
        
        # Reverse to chronological order and join
        state = ' [SEP] '.join(text_list[::-1])
        return state
    
    def _predict_search(self, observation: str) -> str:
        """
        Generate search query from observation.
        
        Args:
            observation: Current observation containing instruction
            
        Returns:
            Search action string
        """
        if not self.use_search_model or self.search_model is None:
            # Fallback: extract instruction and search directly
            goal = process_goal(observation)
            return f'search[{goal}]'
        
        # Use BART to generate search query
        goal = process_goal(observation)
        
        with torch.no_grad():
            # Tokenize input
            input_ids = self.bart_tokenizer(goal, return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.device)
            
            # Generate search query
            output = self.search_model.generate(
                input_ids,
                max_length=512,
                num_return_sequences=1,
                num_beams=5
            )
            
            # Decode output
            query = self.bart_tokenizer.batch_decode(
                output.tolist(),
                skip_special_tokens=True
            )[0]
        
        return f'search[{query}]'
    
    def _predict_choice(
        self,
        state: str,
        valid_acts: list,
        image_feat: Optional[torch.Tensor] = None
    ) -> str:
        """
        Select action from valid actions using BERT model.
        
        Args:
            state: Current state observation
            valid_acts: List of valid action strings
            image_feat: Optional image features
            
        Returns:
            Selected action string
        """
        # Prepare batch
        state_encodings = self.tokenizer.encode(process(state), max_length=512)
        action_encodings = self.tokenizer.encode(
            [process(act) for act in valid_acts],
            max_length=512
        )
        
        batch = {
            'state_input_ids': torch.tensor([state_encodings['input_ids']]).to(self.device),
            'state_attention_mask': torch.tensor([state_encodings['attention_mask']]).to(self.device),
            'action_input_ids': torch.tensor(action_encodings['input_ids']).to(self.device),
            'action_attention_mask': torch.tensor(action_encodings['attention_mask']).to(self.device),
            'sizes': torch.tensor([len(valid_acts)]).to(self.device),
        }
        
        # Add image features if enabled
        if self.use_images and image_feat is not None:
            if isinstance(image_feat, list):
                image_feat = torch.tensor(image_feat)
            batch['images'] = image_feat.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.choice_model(**batch)
            logits = outputs.logits[0]  # First (and only) batch item
        
        # Select action based on method
        if self.sampling_method == 'softmax':
            probs = F.softmax(logits, dim=0)
            idx = torch.multinomial(probs, num_samples=1)[0].item()
        else:  # greedy
            idx = logits.argmax(0).item()
        
        return valid_acts[idx]


def create_il_policy(**kwargs) -> ILPolicy:
    """
    Factory function to create IL policy with error handling.
    
    Args:
        **kwargs: Arguments to pass to ILPolicy constructor
        
    Returns:
        ILPolicy instance
        
    Raises:
        FileNotFoundError: If models are not downloaded
        ImportError: If required packages are missing
    """
    try:
        return ILPolicy(**kwargs)
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        raise
    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nPlease install baseline dependencies:")
        print("  uv pip install -e .[baseline]")
        print()
        raise


