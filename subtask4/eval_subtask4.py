"""
Evaluation script for Subtask 4 (Evidence Alignment)
Calculates Precision, Recall, and F1 over sentence-level alignment links.
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_KEY_FILE = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "dev", "archehr-qa_key.json")
SUBMISSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission_subtask4_dev.json")

def load_gold_links():
    """Load gold alignment links from the Dev set."""
    if not os.path.exists(DEV_KEY_FILE):
        print(f"❌ Gold file not found: {DEV_KEY_FILE}")
        return None
        
    with open(DEV_KEY_FILE, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    gold_links = set()
    for case in gold_data:
        cid = str(case['case_id'])
        for ans in case.get('clinician_answer_sentences', []):
            aid = str(ans['id'])
            citations = str(ans.get('citations', ''))
            
            if citations.strip():
                # Split and add links
                for ev_id in citations.split(','):
                    ev_id = ev_id.strip()
                    if ev_id:
                        gold_links.add(f"{cid}_{aid}_{ev_id}")
                        
    return gold_links

def load_pred_links(filepath):
    """Load prediction alignment links."""
    if not os.path.exists(filepath):
        print(f"❌ Prediction file not found: {filepath}")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        preds = json.load(f)
        
    pred_links = set()
    for case in preds:
        cid = str(case['case_id'])
        for item in case.get('prediction', []):
            aid = str(item['answer_id'])
            ev_list = item.get('evidence_id', [])
            
            # It could be string or list
            if isinstance(ev_list, str):
                ev_list = [ev_list]
                
            for ev_id in ev_list:
                ev_id = str(ev_id).strip()
                if ev_id:
                    pred_links.add(f"{cid}_{aid}_{ev_id}")
                    
    return pred_links

def main():
    print("\n" + "="*50)
    print("Subtask 4: Evidence Alignment - Evaluation")
    print("="*50)
    
    gold_links = load_gold_links()
    if gold_links is None:
        return
        
    print(f"✅ Loaded {len(gold_links)} gold links.")
    
    pred_links = load_pred_links(SUBMISSION_FILE)
    if pred_links is None:
        return
        
    print(f"✅ Loaded {len(pred_links)} predicted links.")
    
    # Calculate Metrics
    tp = len(gold_links & set(pred_links))
    fp = len(set(pred_links) - gold_links)
    fn = len(gold_links - set(pred_links))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print("ALIGNMENT LINK METRICS")
    print("="*50)
    print(f"True Positives  : {tp}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print("-"*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
