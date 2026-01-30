import numpy as np
from typing import Optional

class MCTSNode:
    def __init__(self, state: int, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0 # Total reward from Player 0's perspective
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def ucb1(self, c: float = 1.414, maximize: bool = True) -> float:
        if self.visits == 0:
            return float("inf") if maximize else float("-inf")
        
        exploitation = self.value / self.visits
        exploration = c * np.sqrt(np.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration if maximize else exploitation - exploration
    
    def backward(self, reward):
        """Pass reward (from Player 0's perspective) up the tree."""
        self.value += reward
        self.visits += 1
        if self.parent:
            self.parent.backward(reward)
 