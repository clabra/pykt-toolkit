#!/usr/bin/env python3
"""
Simple usage example for knowledge evolution extraction.
This shows how to analyze a real student from the dataset.
"""

import sys
sys.path.insert(0, '/workspaces/pykt-toolkit')

from extract_knowledge_evolution import KnowledgeStateEvolutionExtractor
from pykt.datasets import init_dataset4train
import torch
import os

def main():
    """Simple example showing how to analyze a real student."""
    
    print("="*60)
    print("REAL STUDENT ANALYSIS EXAMPLE")
    print("="*60)
    
    # 1. Load dataset to get real student data
    print("📚 Loading dataset...")
    dataset_name = "assist2015"
    model_name = "gainakt2"
    data_config = {
        "assist2015": {
            "dpath": "/workspaces/pykt-toolkit/data/assist2015",
            "num_q": 0,
            "num_c": 100,
            "input_type": ["concepts"],
            "max_concepts": 1,
            "min_seq_len": 3,
            "maxlen": 200,
            "emb_path": "",
            "train_valid_original_file": "train_valid.csv",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1, 2, 3, 4],
            "test_original_file": "test.csv",
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv"
        }
    }
    
    try:
        train_loader, valid_loader = init_dataset4train(
            dataset_name, model_name, data_config, 0, 32
        )
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # 2. Get a real student from the validation set
    print("👥 Getting real student data...")
    for batch_idx, batch in enumerate(valid_loader):
        if batch_idx == 0:  # Use first batch
            # Get the first student with a reasonable sequence length
            for i in range(batch['cseqs'].size(0)):
                questions = batch['cseqs'][i]
                responses = batch['rseqs'][i]
                mask = batch['masks'][i]
                
                # Only analyze students with some interactions
                valid_length = mask.sum().item()
                if valid_length > 5:  # At least 5 interactions
                    # Trim to valid length
                    questions = questions[:valid_length]
                    responses = responses[:valid_length]
                    student_id = f"real_student_{batch_idx}_{i}"
                    break
            break
    
    print(f"🎯 Selected student: {student_id}")
    print(f"   Sequence length: {len(questions)}")
    print(f"   Skills practiced: {len(torch.unique(questions))}")
    print(f"   Accuracy: {responses.float().mean():.2%}")
    
    # 3. Model configuration (must match your trained model)
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
    
    # 4. Initialize the knowledge extractor
    print("🧠 Loading trained model...")
    model_path = "saved_model/gainakt2_enhanced_auc_0.7253/model.pth"
    
    try:
        extractor = KnowledgeStateEvolutionExtractor(
            model_path=model_path,
            model_config=model_config
        )
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Model not found at: {model_path}")
        print("   Make sure you've trained the model first!")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # 5. Extract knowledge evolution
    print("🔍 Extracting knowledge state evolution...")
    evolution_data = extractor.extract_student_journey(
        questions, responses, student_id=student_id
    )
    
    # 6. Generate learning analytics report
    print("📊 Generating learning analytics...")
    report = extractor.generate_learning_report(
        evolution_data,
        save_path=f"{student_id}_report.json"
    )
    
    # 7. Create visualization
    print("🎨 Creating visualization...")
    extractor.visualize_mastery_evolution(
        evolution_data,
        save_path=f"{student_id}_evolution.png"
    )
    
    # 8. Print summary
    print("\n" + "="*50)
    print("📈 ANALYSIS RESULTS")
    print("="*50)
    print(f"Student ID: {report['student_id']}")
    print(f"Total Interactions: {report['total_interactions']}")
    print(f"Skills Practiced: {report['skills_practiced']}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
    print(f"Average Final Mastery: {report['avg_final_mastery']:.3f}")
    print(f"Mastery Growth: {report['mastery_growth']:+.3f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   📊 {student_id}_report.json")
    print(f"   📈 {student_id}_evolution.png")
    
    # 9. Show skill-specific insights
    print(f"\n🎯 Top Skills by Final Mastery:")
    skill_data = []
    for skill_name, skill_info in report['skill_breakdown'].items():
        skill_data.append((skill_name, skill_info['final_mastery'], skill_info['interactions_count']))
    
    # Sort by final mastery
    skill_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (skill_name, mastery, interactions) in enumerate(skill_data[:5]):
        print(f"   {i+1}. {skill_name}: {mastery:.3f} mastery ({interactions} interactions)")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("\n💡 Next steps:")
        print("   - Open the PNG file to see the evolution visualization")
        print("   - Check the JSON file for detailed analytics")
        print("   - Modify this script to analyze different students")
        print("   - Use the batch analysis example for multiple students")
    else:
        print("\n⚠️  Analysis failed. Check error messages above.")