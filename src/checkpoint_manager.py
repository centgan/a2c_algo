"""
Checkpoint Manager for A2C Training

Handles saving and loading training state for resuming after Lightning.ai session timeouts.
"""

import json
import pickle
import os
from datetime import datetime
from pathlib import Path
import numpy as np


class CheckpointManager:
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.training_state_file = self.checkpoint_dir / 'training_state.json'
        self.learner_state_file = self.checkpoint_dir / 'learner_state.pkl'
        self.agent_progress_file = self.checkpoint_dir / 'agent_progress.json'
    
    def save_training_state(self, num_agents, start_training, end_training, instrument, indicators):
        """Save overall training configuration"""
        state = {
            'num_agents': num_agents,
            'start_training': start_training.strftime('%Y-%m-%d'),
            'end_training': end_training.strftime('%Y-%m-%d'),
            'instrument': instrument,
            'indicators': indicators,
            'last_saved': datetime.now().isoformat(),
        }
        with open(self.training_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_training_state(self):
        """Load overall training configuration"""
        if not self.training_state_file.exists():
            return None
        with open(self.training_state_file, 'r') as f:
            return json.load(f)
    
    def save_learner_state(self, update_count, best_avg_reward, reward_history):
        """Save learner-specific state"""
        state = {
            'update_count': update_count,
            'best_avg_reward': float(best_avg_reward),
            'reward_history': [float(r) for r in reward_history[-1000:]],  # Keep last 1000
            'last_saved': datetime.now().isoformat(),
        }
        with open(self.learner_state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_learner_state(self):
        """Load learner-specific state"""
        if not self.learner_state_file.exists():
            return None
        with open(self.learner_state_file, 'rb') as f:
            return pickle.load(f)
    
    def save_agent_progress(self, agent_id, completed_loops, current_loop_start=None):
        """Save progress for a specific agent"""
        # Load existing progress
        if self.agent_progress_file.exists():
            with open(self.agent_progress_file, 'r') as f:
                all_progress = json.load(f)
        else:
            all_progress = {}
        
        # Update this agent's progress
        all_progress[str(agent_id)] = {
            'completed_loops': [
                {'start': loop[0].strftime('%Y-%m-%d'), 'end': loop[1].strftime('%Y-%m-%d')}
                for loop in completed_loops
            ],
            'current_loop_start': current_loop_start.strftime('%Y-%m-%d') if current_loop_start else None,
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(self.agent_progress_file, 'w') as f:
            json.dump(all_progress, f, indent=2)
    
    def load_agent_progress(self, agent_id):
        """Load progress for a specific agent"""
        if not self.agent_progress_file.exists():
            return None
        
        with open(self.agent_progress_file, 'r') as f:
            all_progress = json.load(f)
        
        return all_progress.get(str(agent_id))
    
    def get_resume_info(self):
        """Get information about whether we should resume and from where"""
        training_state = self.load_training_state()
        learner_state = self.load_learner_state()
        
        if training_state is None:
            return {
                'should_resume': False,
                'message': 'No checkpoint found. Starting fresh training.'
            }
        
        return {
            'should_resume': True,
            'training_state': training_state,
            'learner_state': learner_state,
            'message': f"Checkpoint found from {training_state['last_saved']}. "
                      f"Learner at update {learner_state['update_count'] if learner_state else 0}."
        }
    
    def checkpoint_exists(self):
        """Check if any checkpoint exists"""
        return self.training_state_file.exists()
