"""
Evaluation script for Subtask 2 (Evidence Identification)
Compares different approaches: BM25, Hybrid, and LLM-only.
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DEV set is split between dev/ (1-20) and test/ (21-120) folders in v1.4
GOLD_FILES = [
    os.path.join(PROJECT_ROOT, "data", "v1.4", "dev", "archehr-qa_key.json"),
    os.path.join(PROJECT_ROOT, "data", "v1.4", "test", "archehr-qa_key.json")
]

def load_gold_labels():
    """Load gold evidence labels from all available key files."""
    gold_map = {}
    
    for gold_file in GOLD_FILES:
        if not os.path.exists(gold_file):
            print(f"⚠️ Warning: Gold file not found: {gold_file}")
            continue
            
        print(f"📄 Loading gold labels from: {gold_file}")
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        
        for g in gold_data:
            if 'answers' not in g:
                continue
                
            cid = str(g['case_id'])
            # Strict: Essential only
            essential = set([a['sentence_id'] for a in g['answers'] if a['relevance'] == 'essential'])
            # Lenient: Essential + Supplementary
            lenient = set([a['sentence_id'] for a in g['answers'] if a['relevance'] in ['essential', 'supplementary']])
            
            gold_map[cid] = {
                'strict': essential,
                'lenient': lenient
            }
    
    print(f"✅ Total Gold Cases Loaded: {len(gold_map)}")
    return gold_map

def load_predictions(filepath):
    """Load prediction file."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    pred_map = {}
    for p in preds:
        pred_map[str(p['case_id'])] = set(p['prediction'])
    
    return pred_map

def evaluate(predictions, gold_map, mode='strict'):
    """Calculate Precision, Recall, F1."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for case_id, pred_set in predictions.items():
        if case_id not in gold_map:
            continue
        
        gold_set = gold_map[case_id][mode]
        
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

def find_submission_files():
    """Auto-detect all Subtask 2 submission files."""
    possible_files = []
    
    # Check root-level files
    patterns = [
        "submission_subtask2_dev*.json",
        "submission_subtask2_test*.json",
    ]
    
    for pattern in patterns:
        import glob
        matches = glob.glob(os.path.join(PROJECT_ROOT, pattern))
        for match in matches:
            name = os.path.basename(match).replace('.json', '').replace('submission_subtask2_', '')
            possible_files.append((name, match))
    
    # Check subtask_2/dev/ and subtask_2/test/
    for mode in ['dev', 'test']:
        sub_path = os.path.join(PROJECT_ROOT, 'script', 'subtask_2', mode, 'submission.json')
        if os.path.exists(sub_path):
            possible_files.append((f"subtask_2/{mode}", sub_path))
    
    return possible_files

def main():
    print("\n" + "="*50)
    print("Subtask 2: Evidence Identification - Evaluation")
    print("="*50)
    
    # Load gold
    gold_map = load_gold_labels()
    print(f"\nLoaded gold labels for {len(gold_map)} cases")
    
    # Auto-detect files
    available_files = find_submission_files()
    
    if not available_files:
        print("❌ No submission files found!")
        print("   Expected files like: submission_subtask2_dev.json")
        return
    
    print("\n📁 Available Submission Files:")
    for idx, (name, path) in enumerate(available_files, 1):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {idx}. {exists} {name}")
    
    print(f"  {len(available_files) + 1}. Evaluate ALL")
    
    choice = input(f"\nSelect file(s) to evaluate (1-{len(available_files) + 1}): ").strip()
    
    files = {}
    try:
        choice_num = int(choice)
        if choice_num == len(available_files) + 1:
            # Evaluate all
            files = {name: path for name, path in available_files}
        elif 1 <= choice_num <= len(available_files):
            # Evaluate single file
            name, path = available_files[choice_num - 1]
            files = {name: path}
        else:
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    results = {}
    
    for name, filepath in files.items():
        preds = load_predictions(filepath)
        if preds is None:
            continue
        
        # Strict evaluation
        strict_metrics = evaluate(preds, gold_map, mode='strict')
        # Lenient evaluation
        lenient_metrics = evaluate(preds, gold_map, mode='lenient')
        
        results[name] = {
            'strict': strict_metrics,
            'lenient': lenient_metrics
        }
    
    # Display results
    print("\n" + "="*50)
    print("STRICT EVALUATION (Essential sentences only)")
    print("="*50)
    print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*50)
    
    for name, metrics in results.items():
        m = metrics['strict']
        print(f"{name:<25} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    print("\n" + "="*50)
    print("LENIENT EVALUATION (Essential + Supplementary)")
    print("="*50)
    print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*50)
    
    for name, metrics in results.items():
        m = metrics['lenient']
        print(f"{name:<25} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    # Find best
    best_method = max(results.items(), key=lambda x: x[1]['strict']['f1'])
    print("\n" + "="*50)
    print(f"✅ Best Method (Strict F1): {best_method[0]}")
    print(f"   F1: {best_method[1]['strict']['f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
