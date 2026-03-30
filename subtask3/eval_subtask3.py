import json
import os
import evaluate
import numpy as np
from bert_score import score as bert_score

# --- Files ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Gold files for DEV set (1-120)
GOLD_FILES = [
    os.path.join(PROJECT_ROOT, "data", "v1.4", "dev", "archehr-qa_key.json"),
    os.path.join(PROJECT_ROOT, "data", "v1.4", "test", "archehr-qa_key.json")
]

def find_submission_files():
    """Auto-detect all Subtask 3 submission files."""
    possible_files = []
    
    # Check root-level files
    import glob
    patterns = [
        "submission_subtask3_dev*.json",
        "submission_subtask3_test*.json",
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(PROJECT_ROOT, pattern))
        for match in matches:
            name = os.path.basename(match).replace('.json', '').replace('submission_subtask3_', '')
            possible_files.append((name, match))
    
    
    # Check subtask_3/dev/, subtask_3/test/, and subtask_3/test-2026/
    for mode in ['dev', 'test', 'test-2026']:
        sub_path = os.path.join(PROJECT_ROOT, 'script', 'subtask_3', mode, 'submission.json')
        if os.path.exists(sub_path):
            possible_files.append((f"subtask_3/{mode}", sub_path))
    
    return possible_files

def load_data(pred_file):
    if not os.path.exists(pred_file):
        print(f"❌ Prediction file not found: {pred_file}")
        return None

    # Load All Gold Data
    gold_data = {}
    for gold_file in GOLD_FILES:
        if not os.path.exists(gold_file):
            print(f"⚠️ Warning: Gold file not found: {gold_file}")
            continue
            
        print(f"📄 Loading gold data from: {gold_file}")
        with open(gold_file, 'r', encoding='utf-8') as f:
            golds = json.load(f)
            
        for g in golds:
            cid = str(g['case_id'])
            gold_data[cid] = {
                "ref": g.get('clinician_answer_without_citations', ""),
                "src": g.get('clinician_interpreted_question', "") # Source để tính SARI
            }
    
    print(f"✅ Total Gold Cases Loaded: {len(gold_data)}")

    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
        
    pred_map = {str(p['case_id']): p['prediction'] for p in preds}
    
    common_ids = sorted(list(set(pred_map.keys()) & set(gold_data.keys())), key=lambda x: int(x))
    
    p_list = [pred_map[cid] for cid in common_ids]
    g_list = [gold_data[cid]["ref"] for cid in common_ids]
    s_list = [gold_data[cid]["src"] for cid in common_ids]
    
    return p_list, g_list, s_list, common_ids

def truncate_to_75_words(text):
    """Quy tắc chính thức: Truncate về 75 từ đầu tiên."""
    if not text: return ""
    words = text.split()
    return " ".join(words[:75])

def evaluate_subtask3(pred_file):
    data = load_data(pred_file)
    if not data: return
    preds_raw, references, sources, ids = data

    # 1. Apply Truncation
    preds = [truncate_to_75_words(p) for p in preds_raw]
    
    results = {}

    # --- ROUGE ---
    print("Calculating ROUGE...")
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=preds, references=references)
    results.update(rouge_results)

    # --- BLEU (SacreBLEU style thường được dùng trong benchmark) ---
    print("Calculating BLEU...")
    bleu = evaluate.load('bleu')
    # BLEU theo PDF thường là sacrebleu hoặc tương đương
    bleu_results = bleu.compute(predictions=preds, references=[[r] for r in references])
    results['bleu'] = bleu_results['bleu']
    
    # --- SARI (Trang 7 PDF nhắc đến) ---
    # SARI = (Keep + Add + Delete) / 3. Cần Source, Prediction, Reference.
    print("Calculating SARI...")
    sari = evaluate.load('sari')
    try:
        sari_results = sari.compute(sources=sources, predictions=preds, references=[[r] for r in references])
        results['sari'] = sari_results['sari']
    except:
        results['sari'] = 0.0

    # --- BERTScore ---
    print("Calculating BERTScore (roberta-large)...")
    # PDF footnote 6 thường ám chỉ roberta-large
    P, R, F1 = bert_score(preds, references, lang="en", model_type="roberta-large", verbose=False)
    results['bert_f1'] = F1.mean().item()

    # --- Summary ---
    print("\n" + "="*30)
    print("OFFICIAL METRICS REPORT")
    print("="*30)
    print(f"ROUGE-L:   {results.get('rougeL', 0):.4f}")
    print(f"BLEU:      {results.get('bleu', 0):.4f}")
    print(f"SARI:      {results.get('sari', 0):.4f}")
    print(f"BERTScore: {results.get('bert_f1', 0):.4f}")
    print("-" * 30)
    print("Note: AlignScore & MEDCON require specialized model checkpoints.")
    print("="*30)

    # Print Samples for Comparison
    print("\n" + "="*30)
    print("QUALITATIVE COMPARISON (First 3 Samples)")
    print("="*30)
    for i in range(min(3, len(ids))):
        print(f"\n[Case {ids[i]}]")
        print(f"Word count (Pred): {len(preds[i].split())}")
        print(f"Gold: {references[i]}")
        print(f"Pred: {preds[i]}")
        print("-" * 15)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Subtask 3: Answer Generation - Evaluation")
    print("="*50)
    
    # Auto-detect files
    available_files = find_submission_files()
    
    if not available_files:
        print("❌ No submission files found!")
        print("   Expected files like: submission_subtask3_dev.json")
        exit(1)
    
    print("\n📁 Available Submission Files:")
    for idx, (name, path) in enumerate(available_files, 1):
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {idx}. {exists} {name}")
    
    choice = input(f"\nSelect file to evaluate (1-{len(available_files)}): ").strip()
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(available_files):
            name, pred_file = available_files[choice_num - 1]
            print(f"\n📊 Evaluating: {name}")
            print(f"   File: {pred_file}\n")
            evaluate_subtask3(pred_file)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input.")