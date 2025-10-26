"""
LLM-based Policy using GPT-3.5 Turbo via LiteLLM

This policy uses a language model to decide the next action based on:
- The current observation
- Available actions
- The shopping instruction/goal
- Past action history (provided to LLM for context)

Features:
- Action History: Tracks and shares past actions with the LLM
- LLM learns to avoid repetition by seeing its action history
- Caching: Uses LiteLLM's disk cache to reduce API costs (disabled after first step)
- Cost optimization: Caches only initial step, then provides fresh context-aware responses
"""
import os
import re
import litellm
from litellm import completion
from litellm.caching import Cache


class LLMPolicy:
    """
    LLM-based policy that uses GPT-3.5 Turbo to decide actions.
    
    Action History:
    ---------------
    - Tracks all actions taken during an episode
    - Includes last 10 actions in the prompt to the LLM
    - Lets the LLM learn from its past actions and avoid repetition
    - The LLM decides whether to try something different
    
    Caching:
    --------
    This policy uses LiteLLM's disk cache for the first action only.
    After the first step, caching is disabled because:
    - Prompts include unique action history
    - Fresh responses are needed for context-aware decisions
    - This balances API cost optimization with effective exploration
    
    The cache is stored in .cache/litellm_cache/ directory and persists across runs.
    """
    
    # Class-level cache tracking (dict to track per-model initialization)
    _cache_initialized = {}
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, enable_cache=True, cache_dir=".cache/litellm_cache"):
        """
        Initialize the LLM policy.
        
        Args:
            model: Model name to use (default: gpt-3.5-turbo)
            temperature: Temperature for generation (default: 0.7)
            enable_cache: Whether to enable disk caching (default: True)
            cache_dir: Base directory for disk cache (default: .cache/litellm_cache)
        """
        self.model = model
        # GPT-5 models only support temperature=1
        if 'gpt-5' in model.lower():
            self.temperature = 1.0
            if temperature != 1.0:
                print(f"⚠️  Note: {model} only supports temperature=1.0, overriding temperature={temperature}")
        else:
            self.temperature = temperature
        self.instruction = None
        self.conversation_history = []
        self.enable_cache = enable_cache
        self.action_history = []  # Track all actions taken
        
        # Verify API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before using LLMPolicy."
            )
        
        # Initialize cache with model-specific directory
        if enable_cache and model not in LLMPolicy._cache_initialized:
            # Create model-specific cache directory to avoid cross-contamination
            model_cache_dir = f"{cache_dir}/{model.replace('/', '_')}"
            litellm.cache = Cache(type="disk", disk_cache_dir=model_cache_dir)
            LLMPolicy._cache_initialized[model] = model_cache_dir
            print(f"💾 LiteLLM disk cache enabled for {model} at: {model_cache_dir}")
        elif enable_cache and model in LLMPolicy._cache_initialized:
            # Reinitialize cache for this model
            model_cache_dir = LLMPolicy._cache_initialized[model]
            litellm.cache = Cache(type="disk", disk_cache_dir=model_cache_dir)
    
    def reset(self):
        """Reset policy state for a new episode."""
        self.instruction = None
        self.conversation_history = []
        self.action_history = []
    
    @classmethod
    def clear_cache(cls):
        """
        Clear the disk cache.
        
        This removes all cached LLM responses from disk, forcing fresh API calls.
        Useful for testing or when you want to ensure fresh responses.
        """
        if cls._cache_initialized and litellm.cache is not None:
            try:
                litellm.cache.flush()
                print("🗑️  Cache cleared successfully")
            except AttributeError:
                # If flush() doesn't exist, try to delete cache files
                import shutil
                cache_dir = ".cache/litellm_cache"
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print(f"🗑️  Cache directory {cache_dir} deleted")
                # Reinitialize cache
                litellm.cache = Cache(type="disk", disk_cache_dir=cache_dir)
        else:
            print("ℹ️  Cache not initialized, nothing to clear")
    
    def forward(self, observation: str, available_actions: dict) -> str:
        """
        Generate action using LLM based on observation and available actions.
        
        Args:
            observation: Current page observation
            available_actions: Dict with 'has_search_bar' and 'clickables'
            
        Returns:
            Action string in format: action_name[action_arg]
        """
        # Extract instruction if at start
        if 'Instruction:' in observation and not self.instruction:
            self._extract_instruction(observation)
        
        # Build the prompt (includes action history)
        prompt = self._build_prompt(observation, available_actions)
        
        # Get LLM response - NO FALLBACK, let errors propagate
        # Disable cache if we have action history (prompt is unique now)
        use_cache = self.enable_cache and not self.action_history
        
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
            max_tokens=150,
            caching=use_cache  # Enable caching only for first step
        )
        
        # Check if response was from cache
        if self.enable_cache and hasattr(response, '_hidden_params'):
            cache_hit = response._hidden_params.get('cache_hit', False)
            if cache_hit:
                print("💾 Cache hit - using cached response")
        
        action = response.choices[0].message.content.strip()
        
        # Clean up the action (remove quotes, extra whitespace, etc.)
        action = action.strip('"').strip("'").strip()
        
        # Validate and fix action format if needed
        action = self._validate_action(action, available_actions)
        
        # Store action in history
        self.action_history.append(action)
        
        return action
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines the agent's role."""
        return """You are a shopping assistant agent navigating the WebShop environment.

Your goal is to find and purchase products that match the given instruction.

CRITICAL: Every action MUST follow one of these EXACT formats:
- search[query] - Search for products with the given query
- click[element] - Click on an element (button name or product ID)

NO OTHER FORMAT IS VALID. Your entire response must be a single action in one of these two formats.

Important rules:
1. Read the instruction carefully to understand what to buy (size, color, options, price, etc.)
2. Use search[query] to find relevant products with targeted queries
3. Use click[B0XXXXXXX] to click on product IDs (starting with B0...)
4. **CRITICAL**: When on a product page, you MUST click on the correct OPTIONS before buying:
   - Look for size, color, pack/count options in the available clickables
   - Match these options to what the instruction asks for
   - Use click[option] for each required option (e.g., click[5x-large], click[mossy oak country dna], click[6 pack])
   - Only use click[buy now] AFTER selecting all required options
5. Check that the price matches the instruction requirement
6. NEVER repeat the same action if it didn't work - try something different
7. To go back to search results, use click[back to search]
8. Make progress - don't get stuck repeating actions

RESPONSE FORMAT REQUIREMENTS:
- Your response must be EXACTLY one action
- Must start with either "search[" or "click["
- Must end with "]"
- No explanations, no extra text, just the action

Valid examples:
search[men's t-shirt long sleeve]
click[B00O30JLDK]
click[5x-large big]
click[mossy oak country dna]
click[back to search]
click[buy now]

Invalid examples (DO NOT USE):
back to search
search for men's t-shirt
click on the buy button
I will search[men's t-shirt]"""
    
    def _build_prompt(self, observation: str, available_actions: dict) -> str:
        """Build the prompt for the LLM."""
        prompt_parts = []
        
        # Add instruction if available with highlighted requirements
        if self.instruction:
            prompt_parts.append(f"Shopping Instruction: {self.instruction}")
            
            # Extract and highlight key requirements
            requirements = self._extract_requirements(self.instruction)
            if requirements:
                prompt_parts.append("")
                prompt_parts.append("Key requirements to match:")
                for req in requirements:
                    prompt_parts.append(f"  - {req}")
            
            prompt_parts.append("")
        
        # Add recent action history so LLM can avoid repetition
        if self.action_history:
            recent_actions = self.action_history[-10:]  # Last 10 actions
            prompt_parts.append("Your recent actions:")
            for i, action in enumerate(recent_actions, 1):
                prompt_parts.append(f"  {i}. {action}")
            prompt_parts.append("")
        
        # Add observation (cleaned up)
        obs_clean = self._clean_observation(observation)
        prompt_parts.append(f"Current Page:\n{obs_clean}")
        prompt_parts.append("")
        
        # Add available actions
        if available_actions.get('has_search_bar'):
            prompt_parts.append("Available actions:")
            prompt_parts.append("- You can use: search[your query]")
        
        clickables = available_actions.get('clickables', [])
        if clickables:
            prompt_parts.append("- You can click on:")
            # Limit to first 20 clickables to avoid token limits
            for clickable in clickables[:20]:
                prompt_parts.append(f"  • {clickable}")
            if len(clickables) > 20:
                prompt_parts.append(f"  ... and {len(clickables) - 20} more")
        
        prompt_parts.append("")
        
        # Add hint if on product page
        if 'buy now' in [c.lower() for c in clickables]:
            prompt_parts.append("Note: You're on a product page. Select all required options before clicking 'Buy Now'.")
            prompt_parts.append("")
        
        prompt_parts.append("What action should I take next?")
        prompt_parts.append("(Respond with ONLY the action in format: search[query] or click[element])")
        
        return "\n".join(prompt_parts)
    
    def _clean_observation(self, observation: str) -> str:
        """Clean observation for better readability."""
        # Split by [SEP] and clean up
        parts = observation.split('[SEP]')
        cleaned = []
        
        for part in parts:
            part = part.strip()
            if part and part not in cleaned:  # Remove duplicates
                cleaned.append(part)
        
        return '\n'.join(cleaned[:15])  # Limit to first 15 parts to save tokens
    
    def _extract_instruction(self, observation: str):
        """Extract the instruction text."""
        match = re.search(
            r'Instruction:\s*(?:\[SEP\]\s*)?(.+?)(?:\[SEP\]|Back to Search|Search|$)', 
            observation, 
            re.IGNORECASE | re.DOTALL
        )
        if match:
            self.instruction = match.group(1).strip()
    
    def _extract_requirements(self, instruction: str) -> list:
        """Extract key requirements from the instruction for emphasis."""
        requirements = []
        instruction_lower = instruction.lower()
        
        # Extract size
        size_patterns = [
            r'\bsize:\s*(\S+(?:\s+\S+)?)',
            r'\b(\d+(?:\.\d+)?)\s*(?:inch|inches|oz|ounce|cm|mm)\b',
            r'\b(x+\s*-?\s*small|x+\s*-?\s*large|small|medium|large|xx+l|x+l|xs)\b'
        ]
        for pattern in size_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                requirements.append(f"Size: {match.group(1)}")
                break
        
        # Extract color
        color_match = re.search(r'\bcolor:\s*([^,]+?)(?:,|\band\b|$)', instruction_lower)
        if color_match:
            requirements.append(f"Color: {color_match.group(1).strip()}")
        
        # Extract pack/count
        pack_patterns = [
            r'\b(?:pack|count):\s*(\S+)',
            r'\b(\d+)\s*(?:-?\s*pack|-?\s*count)\b'
        ]
        for pattern in pack_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                requirements.append(f"Pack/Count: {match.group(1)}")
                break
        
        # Extract price
        price_match = re.search(r'price\s+(?:lower|less)\s+than\s+(\d+(?:\.\d+)?)', instruction_lower)
        if price_match:
            requirements.append(f"Max price: ${price_match.group(1)}")
        
        return requirements
    
    def _validate_action(self, action: str, available_actions: dict) -> str:
        """Validate and fix action format if needed."""
        # Check if action follows the correct format
        if not re.match(r'^(search|click)\[.+\]$', action, re.IGNORECASE):
            # Try to extract action from common patterns
            if 'search' in action.lower():
                # Extract search query
                match = re.search(r'search[:\s]*\[?([^\]]+)\]?', action, re.IGNORECASE)
                if match:
                    query = match.group(1).strip()
                    return f"search[{query}]"
            elif 'click' in action.lower():
                # Extract click target
                match = re.search(r'click[:\s]*\[?([^\]]+)\]?', action, re.IGNORECASE)
                if match:
                    target = match.group(1).strip()
                    return f"click[{target}]"
            
            # If we can't parse it, raise an error - NO FALLBACK
            raise ValueError(f"Invalid action format from LLM: '{action}'. Expected 'search[query]' or 'click[element]'")
        
        return action

