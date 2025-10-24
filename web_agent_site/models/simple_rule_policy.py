"""
Simple Rule-based Policy - Baseline from WebShop Paper

This implements the simple heuristic baseline from the paper:
1. Search with the full instruction text
2. Click the first product result
3. Buy immediately

This baseline achieves 9.6% success rate in the paper.
"""
import re


class SimpleRulePolicy:
    """
    Simple rule-based policy that mimics the paper's baseline:
    - Search the instruction
    - Click first product
    - Buy it
    """
    
    def __init__(self):
        self.instruction = None
        self.has_searched = False
        self.clicked_product = False
    
    def reset(self):
        """Reset policy state for a new episode."""
        self.instruction = None
        self.has_searched = False
        self.clicked_product = False
    
    def forward(self, observation: str, available_actions: dict) -> str:
        """
        Generate action based on simple heuristic.
        
        Args:
            observation: Current page observation
            available_actions: Dict with 'has_search_bar' and 'clickables'
            
        Returns:
            Action string
        """
        # Extract instruction if at start
        if 'Instruction:' in observation and not self.has_searched:
            self._extract_instruction(observation)
        
        # If search bar available and haven't searched yet, search the instruction
        if available_actions.get('has_search_bar') and not self.has_searched:
            self.has_searched = True
            if self.instruction:
                # Clean up instruction for search
                query = self.instruction.replace('find me', '').replace('i need', '').strip()
                # Limit to first 10 words to avoid too long queries
                query = ' '.join(query.split()[:10])
                return f'search[{query}]'
            return 'search[product]'
        
        clickables = available_actions.get('clickables', [])
        
        if not clickables:
            return 'search[product]'
        
        # If on item page (has "buy now"), just buy
        if 'buy now' in [c.lower() for c in clickables]:
            return 'click[buy now]'
        
        # If on results page, click first product (ASIN format: B0...)
        product_links = [c for c in clickables if c.lower().startswith('b0')]
        if product_links and not self.clicked_product:
            self.clicked_product = True
            return f'click[{product_links[0]}]'
        
        # Default: click first clickable
        return f'click[{clickables[0]}]'
    
    def _extract_instruction(self, observation: str):
        """Extract the instruction text."""
        match = re.search(r'Instruction:\s*(?:\[SEP\]\s*)?(.+?)(?:\[SEP\]|Back to Search|Search|$)', 
                         observation, re.IGNORECASE | re.DOTALL)
        if match:
            self.instruction = match.group(1).strip().lower()

