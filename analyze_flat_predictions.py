#!/usr/bin/env python3
"""Analyze saved p_ref predictions to understand why they're flat"""

import numpy as np

# Load p_ref predictions
pred_file = 'experiments/20260119_075752_gtransformer_quicktest_timevarying_337220/qid_test_question_predictions_reference.txt'

print("Analyzing p_ref predictions...\n")

with open(pred_file, 'r') as f:
    lines = f.readlines()[1:]  # Skip header

# Analyze first 10 sequences
for i, line in enumerate(lines[:10]):
    parts = line.strip().split('\t')
    if len(parts) < 8:
        continue
        
    qid_str = parts[2]  # questions
    cid_str = parts[3]  # concepts
    pred_str = parts[4]  # concept_preds
    
    questions = [int(x) for x in qid_str.split(',')]
    concepts = [int(x) for x in cid_str.split(',')]
    preds = [float(x) for x in pred_str.split(',')]
    
    print(f"Sequence {i+1}:")
    print(f"  Questions: {questions}")
    print(f"  Concepts: {concepts}")
    print(f"  Predictions: {[f'{p:.4f}' for p in preds]}")
    print(f"  Pred std: {np.std(preds):.6f}")
    
    # Check if all predictions are identical
    if len(set(preds)) == 1:
        print(f"  ⚠️  ALL PREDICTIONS IDENTICAL!")
    elif np.std(preds) < 0.001:
        print(f"  ⚠️  PREDICTIONS NEARLY FLAT (std < 0.001)")
        
    # Check if questions or concepts are repeated
    if len(set(questions)) == 1:
        print(f"  → Same question repeated {len(questions)} times")
    if len(set(concepts)) > 1:
        print(f"  → Different concepts: {len(set(concepts))} unique")
    else:
        print(f"  → Same concept repeated")
    
    print()
