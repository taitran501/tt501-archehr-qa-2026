"""
Hybrid Evidence Identification for Subtask 2
Combines BM25 retrieval + LLM reranking for better precision/recall balance.
"""

import json
import os
import xml.etree.ElementTree as ET
from rank_bm25 import BM25Okapi
import numpy as np

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "v1.4", "dev", "archehr-qa_key.json")

def get_dataset_path(mode="dev"):
    """Get path(s) to XML file(s) based on mode."""
    if mode == "test":
        # Official Test-2026 set (Cases 121-167)
        return [os.path.join(PROJECT_ROOT, "data", "v1.4", "test-2026", "archehr-qa.xml")]
    
    # Official Dev set is split: 1-20 in dev/, 21-120 in test/
    return [
        os.path.join(PROJECT_ROOT, "data", "v1.4", "dev", "archehr-qa.xml"),
        os.path.join(PROJECT_ROOT, "data", "v1.4", "test", "archehr-qa.xml")
    ]

def parse_xml_case(case_elem):
    """Extract case data from XML element."""
    case_id = case_elem.get('id')
    
    # Get questions
    patient_q = case_elem.find('patient_narrative')
    patient_question = patient_q.text.strip() if patient_q is not None and patient_q.text else ""
    
    clin_q = case_elem.find('clinician_question')
    clinician_question = clin_q.text.strip() if clin_q is not None and clin_q.text else ""
    
    # Get note sentences
    sentences = {}
    sent_elems = case_elem.findall('.//note_excerpt_sentences/sentence')
    for sent in sent_elems:
        sid = sent.get('id')
        text = sent.text.strip() if sent.text else ""
        sentences[sid] = text
    
    return {
        'case_id': case_id,
        'patient_question': patient_question,
        'clinician_question': clinician_question,
        'sentences': sentences
    }

def load_data(mode="dev"):
    """Load data from XML."""
    xml_paths = get_dataset_path(mode)
    cases = []
    
    for xml_path in xml_paths:
        if not os.path.exists(xml_path):
            print(f"⚠️ Warning: Path not found: {xml_path}")
            continue
            
        print(f"📄 Loading data from: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for case_elem in root.findall('case'):
            cases.append(parse_xml_case(case_elem))
            
    gold_map = {}
    if mode == "dev" and os.path.exists(GOLD_FILE):
        # Load gold labels
        with open(GOLD_FILE, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        
        for g in gold_data:
            cid = str(g['case_id'])
            # Essential sentences only (strict)
            essential = [a['sentence_id'] for a in g['answers'] if a['relevance'] == 'essential']
            gold_map[cid] = set(essential)
            
    return cases, gold_map

def bm25_retrieve(query, sentences, top_k=10, threshold=0.0):
    if not sentences:
        return []
    
    # Prepare corpus
    sent_ids = list(sentences.keys())
    corpus = [sentences[sid].lower().split() for sid in sent_ids]
    
    # Tokenize query
    query_tokens = query.lower().split()
    
    # BM25 ranking
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k above threshold
    ranked_indices = np.argsort(scores)[::-1]
    
    candidates = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] >= threshold:
            candidates.append(sent_ids[idx])
    
    return candidates

def evaluate_predictions(predictions, gold_map):
    """Calculate Precision, Recall, F1 (strict)."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for case_id, pred_set in predictions.items():
        if case_id not in gold_map:
            continue
        
        gold_set = gold_map[case_id]
        pred_set = set(pred_set)
        
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
        'f1': f1
    }

def main():
    print("\n=== Subtask 2: BM25 Baseline Generator ===\n")
    mode = input("Select Dataset (dev/test): ").strip().lower()
    if mode not in ['dev', 'test']: mode = 'dev'
    
    # Load data
    cases, gold_map = load_data(mode)
    print(f"Loaded {len(cases)} cases")
    
    # Config
    if mode == "dev":
        # Run grid search
         configs = [
            {'top_k': 5, 'threshold': 0.0},
            {'top_k': 10, 'threshold': 0.0},
            {'top_k': 15, 'threshold': 0.0},
            {'top_k': 20, 'threshold': 0.0},
        ]
    else:
        # Use high recall config for Test
        configs = [{'top_k': 20, 'threshold': 0.0}]
        
    best_f1 = 0
    best_config = configs[0]
    
    for config in configs:
        predictions = {}
        for case in cases:
            case_id = case['case_id']
            # Combined query: Clinician Question + Patient Question (optional but might help recall)
            # Just usage Clinician Question for now as standard
            query = case['clinician_question'] 
            sentences = case['sentences']
            
            # BM25 retrieval
            pred_ids = bm25_retrieve(query, sentences, **config)
            predictions[case_id] = pred_ids
        
        if gold_map:
            metrics = evaluate_predictions(predictions, gold_map)
            print(f"\nConfig: {config}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_config = config
        else:
            best_config = config # Default to the only config
            
    print(f"\n✅ Selected Config: {best_config}")
    
    # Generate final
    final_predictions = {}
    for case in cases:
        case_id = case['case_id']
        query = case['clinician_question']
        sentences = case['sentences']
        pred_ids = bm25_retrieve(query, sentences, **best_config)
        final_predictions[case_id] = pred_ids
    
    # Save
    submission = [
        {'case_id': cid, 'prediction': list(preds)}
        for cid, preds in final_predictions.items()
    ]
    
    output_file = os.path.join(PROJECT_ROOT, f"submission_subtask2_{mode}_bm25.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
