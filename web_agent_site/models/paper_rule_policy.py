"""
Paper Rule-based Policy for WebShop Environment.

This policy attempts to match the paper's 9.6% success rate baseline.
The paper describes their rule baseline as:
- Searching the instruction text
- Clicking the first result  
- Buying immediately

Our improved version adds:
- Better search query extraction
- Option selection from instruction
- Simple attribute matching
"""
import re
import random
from typing import Dict, List


class PaperRulePolicy:
    """Rule-based policy for WebShop task."""
    
    def __init__(self):
        self.instruction = None
        self.search_terms = []
        self.required_options = {}
        self.has_searched = False
        self.clicked_product = False
        self.options_selected = set()
    
    def reset(self):
        """Reset policy state for a new episode."""
        self.instruction = None
        self.search_terms = []
        self.required_options = {}
        self.has_searched = False
        self.clicked_product = False
        self.options_selected = set()
    
    def forward(self, observation: str, available_actions: Dict) -> str:
        """
        Generate action based on current observation and available actions.
        
        Strategy:
        1. Extract instruction and parse requirements
        2. Search with cleaned instruction text
        3. Click first product
        4. Select options matching instruction
        5. Buy
        
        Args:
            observation: Current page observation (text)
            available_actions: Dict with 'has_search_bar' and 'clickables'
            
        Returns:
            Action string in format "action[argument]"
        """
        # Extract instruction if at start
        if 'Instruction:' in observation and not self.has_searched:
            self._extract_instruction(observation)
        
        # If search bar available and haven't searched yet
        if available_actions.get('has_search_bar') and not self.has_searched:
            self.has_searched = True
            if self.instruction:
                # Use cleaned instruction as search query
                query = ' '.join(self.search_terms[:10])  # Limit to 10 terms
                return f'search[{query}]'
            return 'search[product]'
        
        clickables = available_actions.get('clickables', [])
        if not clickables:
            return 'search[product]'
        
        # If on item page (has "buy now")
        if 'buy now' in [c.lower() for c in clickables]:
            return self._handle_item_page(observation, clickables)
        
        # If on results page, click first product
        product_links = [c for c in clickables if c.lower().startswith('b0')]
        if product_links and not self.clicked_product:
            self.clicked_product = True
            return f'click[{product_links[0]}]'
        
        # Default: click first clickable
        return f'click[{clickables[0]}]'
    
    def _extract_instruction(self, observation: str):
        """Extract and parse the instruction to identify key requirements."""
        # Extract instruction text
        match = re.search(r'Instruction:\s*(?:\[SEP\]\s*)?(.+?)(?:\[SEP\]|Back to Search|Search|$)', 
                         observation, re.IGNORECASE | re.DOTALL)
        
        if match:
            self.instruction = match.group(1).strip().lower()
            
            # Clean instruction for search: remove common filler words
            cleaned = self.instruction.replace('find me', '').replace('i need', '').replace('i want', '')
            
            # Extract all words
            words = re.findall(r'\b\w+\b', cleaned)
            self.search_terms = [w for w in words if len(w) > 2]
            
            # Extract options from instruction
            self._extract_options()
    
    def _extract_options(self):
        """Extract required options from instruction (size, color, pack, etc.)."""
        if not self.instruction:
            return
        
        # Extract size (e.g., "size 7", "7 inch", "medium", "large")
        size_patterns = [
            r'\bsize\s+(\w+(?:\.\d+)?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:inch|inches|oz|ounce|cm|mm|ft|feet)\b',
            r'\b(small|medium|large|xl|xxl|xs)\b'
        ]
        for pattern in size_patterns:
            match = re.search(pattern, self.instruction)
            if match:
                self.required_options['size'] = match.group(1)
                break
        
        # Extract color
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey',
                 'pink', 'purple', 'orange', 'beige', 'navy', 'silver', 'gold', 'cream', 'tan']
        for color in colors:
            if color in self.instruction:
                self.required_options['color'] = color
                break
        
        # Extract pack/count (e.g., "pack of 6", "6 pack", "6 count")
        pack_patterns = [
            r'\b(?:pack of|count of|set of)\s+(\d+)\b',
            r'\b(\d+)\s*(?:pack|count)\b'
        ]
        for pattern in pack_patterns:
            match = re.search(pattern, self.instruction)
            if match:
                self.required_options['pack'] = match.group(1)
                break
    
    
    def _handle_item_page(self, observation: str, clickables: List[str]) -> str:
        """
        Handle item page - select options matching instruction, then buy.
        
        Strategy:
        - Try to select options that match instruction (size, color, pack)
        - Only select each option once (track with options_selected)
        - Once no more matching options, buy
        """
        # Try to select required options that match instruction
        for option_name, option_value in self.required_options.items():
            # Skip if we've already tried to select this option
            if option_name in self.options_selected:
                continue
                
            option_value_lower = str(option_value).lower()
            
            # Look for matching option in clickables
            for clickable in clickables:
                clickable_lower = clickable.lower()
                # Check if this clickable matches our required option
                # Exclude navigation buttons
                if (option_value_lower in clickable_lower and 
                    clickable_lower not in ['buy now', 'description', 'features', 'reviews', 
                                           'back to search', '< prev', 'next >']):
                    self.options_selected.add(option_name)
                    return f'click[{clickable}]'
            
            # Mark this option as "tried" even if not found
            self.options_selected.add(option_name)
        
        # If we've tried all options or have none, buy
        if 'buy now' in [c.lower() for c in clickables]:
            return 'click[buy now]'
        
        # Fallback
        return f'click[{clickables[0]}]'
