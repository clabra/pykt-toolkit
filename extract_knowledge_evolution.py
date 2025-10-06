#!/usr/bin/env python3
"""
Knowledge State Evolution Extractor for GainAKT2Monitored

This script demonstrates how to extract and visualize the evolution of 
knowledge states (skill mastery levels) from the trained GainAKT2Monitored model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, '/workspaces/pykt-toolkit')

from pykt.models.gainakt2_monitored import create_monitored_model
from pykt.datasets import init_dataset4train


class KnowledgeStateEvolutionExtractor:
    """Extract and analyze knowledge state evolution from GainAKT2Monitored."""
    
    def __init__(self, model_path, model_config):
        """
        Initialize the extractor with a trained model.
        
        Args:
            model_path (str): Path to the saved model checkpoint
            model_config (dict): Model configuration parameters
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        self.model = create_monitored_model(model_config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.num_c = model_config['num_c']
        
    def extract_student_journey(self, questions, responses, student_id=None):
        """
        Extract the complete knowledge state evolution for a single student.
        
        Args:
            questions (torch.Tensor): Question sequence [seq_len]
            responses (torch.Tensor): Response sequence [seq_len]
            student_id (str, optional): Student identifier for labeling
            
        Returns:
            dict: Complete knowledge state evolution data
        """
        with torch.no_grad():
            # Ensure proper shapes and device
            if questions.dim() == 1:
                questions = questions.unsqueeze(0)  # Add batch dimension
            if responses.dim() == 1:
                responses = responses.unsqueeze(0)
                
            questions = questions.to(self.device)
            responses = responses.to(self.device)
            
            # Get model outputs with internal states
            outputs = self.model.forward_with_states(
                q=questions, 
                r=responses, 
                qry=None,
                batch_idx=None
            )
            
            # Extract evolution data
            projected_mastery = outputs['projected_mastery'].cpu().numpy()  # [1, seq_len, num_c]
            projected_gains = outputs['projected_gains'].cpu().numpy()      # [1, seq_len, num_c]
            predictions = outputs['predictions'].cpu().numpy()              # [1, seq_len]
            
            # Remove batch dimension
            projected_mastery = projected_mastery[0]  # [seq_len, num_c]
            projected_gains = projected_gains[0]      # [seq_len, num_c]
            predictions = predictions[0]              # [seq_len]
            
            seq_len = projected_mastery.shape[0]
            
            return {
                'student_id': student_id or 'unknown',
                'sequence_length': seq_len,
                'questions': questions[0].cpu().numpy(),
                'responses': responses[0].cpu().numpy(),
                'predictions': predictions,
                'mastery_evolution': projected_mastery,    # [seq_len, num_c]
                'learning_gains': projected_gains,         # [seq_len, num_c]
                'time_steps': np.arange(seq_len)
            }
    
    def get_skill_mastery_trajectory(self, evolution_data, skill_ids=None):
        """
        Get mastery trajectories for specific skills.
        
        Args:
            evolution_data (dict): Output from extract_student_journey()
            skill_ids (list, optional): Specific skills to track. If None, uses practiced skills.
            
        Returns:
            dict: Skill mastery trajectories over time
        """
        if skill_ids is None:
            # Use skills that were actually practiced
            skill_ids = np.unique(evolution_data['questions'])
        
        trajectories = {}
        mastery_evolution = evolution_data['mastery_evolution']
        
        for skill_id in skill_ids:
            if skill_id < self.num_c:  # Valid skill ID
                trajectories[f'skill_{skill_id}'] = {
                    'time_steps': evolution_data['time_steps'],
                    'mastery_levels': mastery_evolution[:, skill_id],
                    'skill_id': skill_id
                }
        
        return trajectories
    
    def get_interaction_effects(self, evolution_data):
        """
        Analyze the effect of each interaction on knowledge states.
        
        Returns:
            pd.DataFrame: Interaction effects analysis
        """
        questions = evolution_data['questions']
        responses = evolution_data['responses']
        gains = evolution_data['learning_gains']
        mastery_before = evolution_data['mastery_evolution']
        
        interactions = []
        
        for t in range(len(questions)):
            skill_id = questions[t]
            response = responses[t]
            
            # Mastery levels
            mastery_current = mastery_before[t, skill_id]
            mastery_prev = mastery_before[t-1, skill_id] if t > 0 else 0.0
            
            # Learning gain for this skill
            learning_gain = gains[t, skill_id]
            
            interactions.append({
                'time_step': t,
                'skill_id': skill_id,
                'response': response,
                'response_label': 'correct' if response == 1 else 'incorrect',
                'mastery_before': mastery_prev,
                'mastery_after': mastery_current,
                'mastery_change': mastery_current - mastery_prev,
                'learning_gain': learning_gain,
                'model_prediction': evolution_data['predictions'][t]
            })
        
        return pd.DataFrame(interactions)
    
    def visualize_mastery_evolution(self, evolution_data, save_path=None, skills_to_show=None):
        """
        Create visualizations of mastery evolution.
        
        Args:
            evolution_data (dict): Student journey data
            save_path (str, optional): Path to save the plot
            skills_to_show (list, optional): Specific skills to visualize
        """
        if skills_to_show is None:
            # Show skills that were practiced
            practiced_skills = np.unique(evolution_data['questions'])
            skills_to_show = practiced_skills[:8]  # Limit to 8 for readability
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Knowledge State Evolution - Student {evolution_data["student_id"]}', fontsize=16)
        
        # 1. Mastery trajectories for practiced skills
        ax1 = axes[0, 0]
        mastery_data = evolution_data['mastery_evolution']
        time_steps = evolution_data['time_steps']
        
        for skill_id in skills_to_show:
            if skill_id < self.num_c:
                ax1.plot(time_steps, mastery_data[:, skill_id], 
                        label=f'Skill {skill_id}', marker='o', alpha=0.7)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Mastery Level')
        ax1.set_title('Skill Mastery Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning gains heatmap
        ax2 = axes[0, 1]
        gains_subset = evolution_data['learning_gains'][:, skills_to_show]
        im = ax2.imshow(gains_subset.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Skill ID')
        ax2.set_title('Learning Gains Heatmap')
        ax2.set_yticks(range(len(skills_to_show)))
        ax2.set_yticklabels([f'Skill {s}' for s in skills_to_show])
        plt.colorbar(im, ax=ax2)
        
        # 3. Performance vs Mastery
        ax3 = axes[1, 0]
        questions = evolution_data['questions']
        responses = evolution_data['responses']
        predictions = evolution_data['predictions']
        
        # Get mastery for practiced skills at interaction time
        interaction_masteries = []
        for t, skill_id in enumerate(questions):
            if skill_id < self.num_c:
                interaction_masteries.append(mastery_data[t, skill_id])
            else:
                interaction_masteries.append(0.0)
        
        # Scatter plot: mastery vs actual performance
        colors = ['red' if r == 0 else 'green' for r in responses]
        ax3.scatter(interaction_masteries, predictions, c=colors, alpha=0.6)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        ax3.set_xlabel('Mastery Level (Model Internal)')
        ax3.set_ylabel('Prediction Probability')
        ax3.set_title('Mastery vs Predictions\\n(Red=Incorrect, Green=Correct)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative mastery growth
        ax4 = axes[1, 1]
        # Average mastery across all practiced skills
        practiced_mask = np.zeros(self.num_c, dtype=bool)
        practiced_mask[skills_to_show] = True
        avg_mastery = mastery_data[:, practiced_mask].mean(axis=1)
        
        ax4.plot(time_steps, avg_mastery, 'b-', linewidth=2, label='Avg Mastery')
        ax4.fill_between(time_steps, 0, avg_mastery, alpha=0.3)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Average Mastery Level')
        ax4.set_title('Cumulative Learning Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_learning_report(self, evolution_data, save_path=None):
        """
        Generate a comprehensive learning analytics report.
        
        Args:
            evolution_data (dict): Student journey data  
            save_path (str, optional): Path to save the report
            
        Returns:
            dict: Learning analytics summary
        """
        interactions_df = self.get_interaction_effects(evolution_data)
        
        # Overall statistics
        practiced_skills = np.unique(evolution_data['questions'])
        final_masteries = evolution_data['mastery_evolution'][-1, practiced_skills]
        
        report = {
            'student_id': evolution_data['student_id'],
            'total_interactions': int(evolution_data['sequence_length']),
            'skills_practiced': int(len(practiced_skills)),
            'overall_accuracy': float(evolution_data['responses'].mean()),
            'avg_final_mastery': float(final_masteries.mean()),
            'mastery_growth': float(final_masteries.mean() - evolution_data['mastery_evolution'][0, practiced_skills].mean()),
            'skill_breakdown': {}
        }
        
        # Per-skill analysis
        for skill_id in practiced_skills:
            skill_interactions = interactions_df[interactions_df['skill_id'] == skill_id]
            
            if len(skill_interactions) > 0:
                report['skill_breakdown'][f'skill_{skill_id}'] = {
                    'interactions_count': int(len(skill_interactions)),
                    'accuracy': float(skill_interactions['response'].mean()),
                    'initial_mastery': float(skill_interactions['mastery_before'].iloc[0]),
                    'final_mastery': float(skill_interactions['mastery_after'].iloc[-1]),
                    'total_learning_gain': float(skill_interactions['learning_gain'].sum()),
                    'avg_learning_gain': float(skill_interactions['learning_gain'].mean())
                }
        
        # Save report if requested
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_path}")
        
        return report


def demo_knowledge_evolution():
    """Demonstration of knowledge state evolution extraction."""
    
    # Model configuration (use your trained model config)
    model_config = {
        'num_c': 100,
        'seq_len': 200,
        'd_model': 512,
        'n_heads': 4,
        'num_encoder_blocks': 4,
        'd_ff': 512,
        'dropout': 0.4,
        'emb_type': 'qid',
        'non_negative_loss_weight': 0.485828,
        'consistency_loss_weight': 0.173548,
        'monitor_frequency': 25
    }
    
    # Path to your trained model
    model_path = "saved_model/gainakt2_enhanced_auc_0.7253/model.pth"
    
    try:
        # Initialize extractor
        extractor = KnowledgeStateEvolutionExtractor(model_path, model_config)
        print("✅ Model loaded successfully!")
        
        # Example: Create synthetic student data (replace with real data loading)
        questions = torch.tensor([12, 35, 67, 12, 89, 35, 45, 67, 12, 89])
        responses = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        
        # Extract knowledge evolution
        print("🔍 Extracting knowledge state evolution...")
        evolution_data = extractor.extract_student_journey(
            questions, responses, student_id="demo_student_001"
        )
        
        # Generate learning report
        print("📊 Generating learning analytics report...")
        report = extractor.generate_learning_report(evolution_data)
        
        print(f"\\n📈 LEARNING ANALYTICS SUMMARY:")
        print(f"Student ID: {report['student_id']}")
        print(f"Total Interactions: {report['total_interactions']}")
        print(f"Skills Practiced: {report['skills_practiced']}")
        print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
        print(f"Average Final Mastery: {report['avg_final_mastery']:.3f}")
        print(f"Mastery Growth: +{report['mastery_growth']:.3f}")
        
        # Visualize evolution
        print("🎨 Creating visualization...")
        extractor.visualize_mastery_evolution(
            evolution_data, 
            save_path="knowledge_evolution_demo.png"
        )
        
        return True
        
    except FileNotFoundError:
        print("❌ Model file not found. Please check the model path.")
        print("Expected path: saved_model/gainakt2_enhanced_auc_0.7253/model.pth")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("KNOWLEDGE STATE EVOLUTION EXTRACTION DEMO")
    print("="*60)
    
    success = demo_knowledge_evolution()
    
    if success:
        print("\\n🎉 Demo completed successfully!")
        print("Check the generated files:")
        print("  - knowledge_evolution_demo.png")
        print("  - Learning analytics printed above")
    else:
        print("\\n⚠️  Demo failed. Check the error messages above.")