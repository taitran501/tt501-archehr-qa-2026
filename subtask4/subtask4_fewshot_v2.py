"""
Subtask 4 v2: Evidence Alignment — Strategy A (Recall-Boosted Prompt)
Key changes vs v1:
  - Improved system prompt with recall-aware instructions
  - Patient question added to user prompt for richer context
  - 4 carefully selected few-shot examples (cases 1, 2, 5, 6: cover single/multi/heavy-multi citation patterns)
  - Mini reasoning annotations in examples to teach WHY citations apply
"""

import json
import os
import time
import requests
import xml.etree.ElementTree as ET
import re

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
BASE_URL = "https://api.x.ai/v1"
MODEL_NAME = "grok-4-1-fast-reasoning"

# Paths
DEV_XML = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "dev", "archehr-qa.xml")
DEV_KEY = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "dev", "archehr-qa_key.json")
TEST1_XML = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test", "archehr-qa.xml")
TEST1_KEY = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test", "archehr-qa_key.json")
TEST2_XML = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test-2026", "archehr-qa.xml")
TEST2_KEY = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test-2026", "archehr-qa_key.json")

# Number of few-shot examples to include (selected from dev set)
# Chosen indices: cases with diverse citation patterns (single, multi, heavy-multi)
FEW_SHOT_CASE_IDS = ["1", "2", "5", "6"]

HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are an expert clinical evidence aligner specialized in grounding answer sentences to clinical note excerpts.

**Task**: For each answer sentence, identify ALL note sentence(s) that provide direct evidential support. Return a JSON array of alignment objects.

**Key Alignment Principles**:
1. An answer sentence often synthesizes or paraphrases information from MULTIPLE note sentences. When this happens, cite ALL contributing note sentences, not just the most obvious one.
2. If an answer sentence makes a claim that combines facts from different note sentences (e.g., diagnosis + treatment outcome), each fact's source note sentence should be cited.
3. A note sentence supports an answer sentence if it provides specific clinical facts, findings, procedures, medications, or outcomes that the answer sentence references — even if the wording differs (paraphrasing is common).
4. If an answer sentence is a general conclusion or world-knowledge statement NOT grounded in any specific note sentence, assign an empty array [].
5. Do NOT over-cite: only cite note sentences containing information actually referenced in the answer sentence. A note sentence that merely discusses a related topic without contributing specific content to the answer should NOT be cited.

**Output Rules**:
1. Return ONLY valid JSON as a list of objects. DO NOT wrap in Markdown blocks, no ```json``` tags.
2. Format: [{"answer_id": "1", "evidence_id": ["2", "5"]}, {"answer_id": "2", "evidence_id": []}]
3. Include an entry for EVERY answer sentence, in order.
4. Alignments are many-to-many: one note sentence can support multiple answer sentences, and one answer sentence can be supported by multiple note sentences."""


def parse_xml_sentences(xml_path):
    """Parse XML to get note sentences and questions for a case."""
    tree = ET.parse(xml_path)
    data = {}
    for case in tree.getroot().findall('case'):
        cid = case.get('id')
        clin_q = case.find('clinician_question').text.strip()
        pat_q = case.find('patient_question').text.strip()
        
        sents = {}
        for s in case.findall('.//note_excerpt_sentences/sentence'):
            sents[s.get('id')] = s.text.strip()
            
        data[cid] = {
            'clinician_question': clin_q,
            'patient_question': pat_q,
            'note_sentences': sents
        }
    return data


def get_few_shot_examples():
    """Load strategically selected examples from Dev set with reasoning annotations."""
    if not os.path.exists(DEV_KEY):
        return ""
    
    with open(DEV_KEY, 'r', encoding='utf-8') as f:
        gold = json.load(f)
        
    xml_data = parse_xml_sentences(DEV_XML)
    
    # Build lookup by case_id
    gold_by_id = {item['case_id']: item for item in gold}
    
    # Reasoning annotations for each example to teach the model WHY
    reasoning_annotations = {
        "1": {
            "1": "Answer mentions 'ERCP was recommended to place a common bile duct stent' → Note [2] describes ERCP placing a CBD stent.",
            "2": "Answer mentions 'drainage of biliary obstruction caused by stones and sludge' → Note [2] states the same.",
            "3": "Answer mentions 'no improvement in liver function, needed repeat ERCP' → Note [6] describes return for re-evaluation due to upward trending LFTs.",
            "4": "Answer mentions 'biliary stent obstructed by stones and sludge' → Note [7] states the stent was acutely obstructed.",
            "5": "Answer mentions 'sphincterotomy' and 'stones removed' → Note [8] describes sphincterotomy and stone removal."
        },
        "2": {
            "1": "Answer mentions 'Lasix for acute diastolic heart failure with shortness of breath and edema' → Note [2] describes acute diastolic heart failure with SOB and edema, Note [5] describes Lasix treatment with transition to torsemide.",
            "2": "Answer mentions '8 liters of fluid to maintain BP' and 'contributed to respiratory failure' → Note [6] mentions 8L on presentation for BP, Note [7] states this contributed to hypoxemic respiratory failure.",
            "3": "Answer mentions 'heart failure treated with Lasix, improvement in SOB and oxygen, small amount of supplemental oxygen' → Note [5] describes Lasix treatment with improvement and remaining on small supplemental O2."
        },
        "5": {
            "1": "Answer says 'chest pain was musculoskeletal, not overdose or cardiac' — this is a conclusion drawn from multiple findings: Note [9] shows musculoskeletal exam findings, Note [10] shows normal EKG, Note [18] shows negative cardiac enzymes, Note [19] shows normal TTE results. ALL contribute to this conclusion.",
            "2": "Answer mentions 'pain with movement and palpation' → Note [9] describes exam findings of reproducible pain.",
            "3": "Answer mentions 'EKG showed no ischemia' and 'blood tests confirmed no heart attack' → Note [10] describes EKG findings, Note [18] describes troponin/enzyme results.",
            "4": "Answer mentions 'TTE was performed to rule out cardiac events' → Note [11] describes TTE being ordered.",
            "5": "Answer mentions 'TTE results were normal' → Note [12] describes normal TTE findings, Note [19] confirms normal cardiac structure.",
            "6": "Answer mentions 'telemetry monitoring, no significant cardiac events' → Note [13] describes telemetry monitoring results."
        },
        "6": {
            "1": "Answer mentions 'Candida infection in blood' → Note [5] describes blood culture growing Candida.",
            "2": "Answer mentions 'sputum test revealed yeast, gram positive cocci, gram negative rods' → Note [13] describes sputum culture results.",
            "3": "Answer lists multiple antibiotics → Note [14] mentions vancomycin, [15] mentions linezolid, [18] mentions amikacin, [19] mentions ambisome, [20] mentions tobramycin. Each antibiotic name comes from a different note sentence.",
            "4": "Answer mentions 'antibiotics did not improve condition, worsening chest x-ray, WBC, respiratory distress' → Note [15] mentions ongoing issues, [17] mentions worsening imaging, [19] mentions continued deterioration, [20] mentions respiratory distress.",
            "5": "Answer mentions 'discussions with family, treatment reduced to comfort measures' → Note [7] describes family meeting, Note [8] describes transition to comfort measures.",
            "6": "Answer mentions 'patient died peacefully' → Note [10] describes patient's death."
        }
    }
    
    examples = ""
    for case_id in FEW_SHOT_CASE_IDS:
        if case_id not in gold_by_id or case_id not in xml_data:
            continue
            
        item = gold_by_id[case_id]
        info = xml_data[case_id]
        
        note_text = ""
        for sid, text in sorted(info['note_sentences'].items(), key=lambda x: int(x[0])):
            note_text += f"[{sid}] {text}\n"
            
        answer_text = ""
        expected_output = []
        reasoning_text = ""
        
        case_reasoning = reasoning_annotations.get(case_id, {})
        
        for ans in item['clinician_answer_sentences']:
            ans_id = ans.get('id')
            ans_txt = ans.get('text')
            answer_text += f"[Answer {ans_id}] {ans_txt}\n"
            citations = ans.get('citations', '')
            if citations:
                ev_ids = [c.strip() for c in citations.split(',')]
            else:
                ev_ids = []
            expected_output.append({"answer_id": ans_id, "evidence_id": ev_ids})
            
            # Add reasoning annotation if available
            if ans_id in case_reasoning:
                reasoning_text += f"  Answer {ans_id}: {case_reasoning[ans_id]}\n"
        
        examples += (
            f"---\n"
            f"**Example Case (ID {case_id}):**\n"
            f"**Patient Question:** {info['patient_question']}\n"
            f"**Clinician Question:** {info['clinician_question']}\n"
            f"**Note Excerpt:**\n{note_text}\n"
            f"**Answer Sentences:**\n{answer_text}\n"
            f"**Reasoning:**\n{reasoning_text}\n"
            f"**Output:** {json.dumps(expected_output)}\n\n"
        )
        
    return examples


def create_batch_requests(key_data, xml_data, examples):
    batch_requests = []
    
    for item in key_data:
        case_id = item['case_id']
        info = xml_data[case_id]
        
        note_text = ""
        for sid, text in sorted(info['note_sentences'].items(), key=lambda x: int(x[0])):
            note_text += f"[{sid}] {text}\n"
            
        answer_text = ""
        for ans in item['clinician_answer_sentences']:
            ans_id = ans.get('id')
            ans_txt = ans.get('text')
            answer_text += f"[Answer {ans_id}] {ans_txt}\n"
            
        user_content = (
            f"{examples}"
            f"---\n"
            f"**Current Case:**\n"
            f"**Patient Question:** {info['patient_question']}\n"
            f"**Clinician Question:** {info['clinician_question']}\n"
            f"**Note Excerpt:**\n{note_text}\n"
            f"**Answer Sentences:**\n{answer_text}\n"
            f"**Output:**"
        )
        
        request = {
            "batch_request_id": case_id,
            "batch_request": {
                "chat_get_completion": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 4000
                }
            }
        }
        batch_requests.append(request)
        
    return batch_requests


def parse_llm_response(content):
    """Robust JSON parsing for LLM response content."""
    prediction_data = []
    
    # Clean markdown wrappers
    if "```json" in content:
        clean_content = content.split("```json")[-1].split("```")[0].strip()
    elif "```" in content:
        clean_content = content.split("```")[1].split("```")[0].strip()
    else:
        clean_content = content.strip()
    
    # Try direct parse
    parsed = False
    try:
        prediction_data = json.loads(clean_content)
        parsed = True
    except Exception:
        # Try adding common missing closures for truncated JSON
        for closure in ["]", "}]", '"}]']:
            try:
                prediction_data = json.loads(clean_content + closure)
                parsed = True
                break
            except Exception:
                pass
    
    # Try finding JSON array in content
    if not parsed:
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                prediction_data = json.loads(content[start_idx:end_idx+1])
                parsed = True
            except Exception:
                pass
    
    # Ultimate regex fallback extracting individual objects
    if not parsed:
        matches = re.findall(r'\{[^{}]*"answer_id"[^{}]*"evidence_id"[^{}]*\}', content)
        for m in matches:
            try:
                prediction_data.append(json.loads(m))
            except Exception:
                pass
    
    return prediction_data


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Subtask 4 v2: Evidence Alignment (Strategy A — Recall-Boosted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subtask4_fewshot_v2.py upload --split test\n"
            "  python subtask4_fewshot_v2.py status --batch-id <BATCH_ID>\n"
            "  python subtask4_fewshot_v2.py download --batch-id <BATCH_ID> --split test\n"
        )
    )
    parser.add_argument("action", choices=["upload", "status", "download"],
                        help="Action to perform")
    parser.add_argument("--split", choices=["dev", "test"], default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--batch-id", dest="batch_id",
                        help="Batch ID (required for status and download)")
    args = parser.parse_args()

    print("\n=== Subtask 4 v2: Evidence Alignment (Strategy A — Recall-Boosted) ===\n")

    # Load data
    print("📖 Loading Data...")
    if not os.path.exists(DEV_XML) or not os.path.exists(DEV_KEY):
        print(f"❌ Dev data files missing.")
        dev_key_data, dev_xml_data = [], {}
    else:
        dev_xml_data = parse_xml_sentences(DEV_XML)
        with open(DEV_KEY, 'r', encoding='utf-8') as f:
            dev_key_data = json.load(f)
            
    # --- Load Test Data ---
    if not os.path.exists(TEST1_KEY) or not os.path.exists(TEST2_KEY):
        print(f"❌ Test data files missing in {os.path.dirname(TEST1_KEY)}")
        test_key_data, test_xml_data = [], {}
    else:
        test_xml_data1 = parse_xml_sentences(TEST1_XML)
        with open(TEST1_KEY, 'r', encoding='utf-8') as f:
            test_key_data1 = json.load(f)
            
        test_xml_data2 = parse_xml_sentences(TEST2_XML)
        with open(TEST2_KEY, 'r', encoding='utf-8') as f:
            test_key_data2 = json.load(f)
            
        test_xml_data = {**test_xml_data1, **test_xml_data2}
        test_key_data = test_key_data1 + test_key_data2
        
    examples = get_few_shot_examples()

    print(f"✅ Loaded {len(dev_key_data)} dev cases and {len(test_key_data)} test cases.")
    print(f"✅ Using {len(FEW_SHOT_CASE_IDS)} few-shot examples (cases: {FEW_SHOT_CASE_IDS})")
    print(f"✅ Model: {MODEL_NAME}")

    if args.action == "upload":
        if args.split == "dev":
            batch_reqs = create_batch_requests(dev_key_data, dev_xml_data, examples)
            bname = f"subtask4_v2_dev_{int(time.time())}"
        else:
            batch_reqs = create_batch_requests(test_key_data, test_xml_data, examples)
            bname = f"subtask4_v2_test_{int(time.time())}"

        print(f"📦 Prepared {len(batch_reqs)} requests.")

        try:
            b_resp = requests.post(
                f"{BASE_URL}/batches",
                headers=HEADERS,
                json={"name": bname},
                timeout=30
            )
            if b_resp.status_code != 200:
                print(f"❌ Error creating batch: {b_resp.text}")
                return

            batch_info = b_resp.json()
            batch_id = batch_info.get('batch_id', batch_info.get('id'))
            print(f"✅ Batch {batch_id} created. Uploading requests...")

            chunk_size = 50
            for i in range(0, len(batch_reqs), chunk_size):
                chunk = batch_reqs[i:i+chunk_size]
                requests.post(
                    f"{BASE_URL}/batches/{batch_id}/requests",
                    headers=HEADERS,
                    json={"batch_requests": chunk},
                    timeout=60
                )

            print(f"🎉 Upload Complete! Batch ID: {batch_id}")

        except Exception as e:
            print(f"❌ Error: {e}")

    elif args.action == "status":
        if not args.batch_id:
            parser.error("--batch-id is required for the status action")
        resp = requests.get(f"{BASE_URL}/batches/{args.batch_id}", headers=HEADERS)
        if resp.status_code == 200:
            info = resp.json()
            print(f"Status: {info.get('status')}")
            counts = info.get('request_counts', {})
            if counts:
                print(f"Counts: {counts}")
        else:
            print(f"❌ Error: {resp.text}")

    elif args.action == "download":
        if not args.batch_id:
            parser.error("--batch-id is required for the download action")

        out_file_name = f"submission_subtask4_{args.split}.json"

        print("Fetching results...")
        results = []
        url = f"{BASE_URL}/batches/{args.batch_id}/results?page_size=100"
        while url:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200:
                print(f"❌ Error fetching page: {resp.text}")
                break
            data = resp.json()
            results.extend(data.get('results', []))
            token = data.get('pagination_token')
            url = f"{BASE_URL}/batches/{args.batch_id}/results?page_size=100&pagination_token={token}" if token else None

        print(f"📥 Fetched {len(results)} results.")

        final_submission = []
        parse_errors = 0
        for res in results:
            cid = res.get('batch_request_id') or res.get('custom_id')
            if not cid:
                continue
            try:
                batch_res = res.get('batch_result', {})
                response = batch_res.get('response', {})
                body = response.get('body', response)
                choices = body.get('choices', [])
                if not choices:
                    choices = body.get('chat_get_completion', {}).get('choices', [])

                if choices:
                    content = choices[0]['message']['content'].strip()
                    prediction_data = parse_llm_response(content)

                    if not prediction_data:
                        print(f"❌ Could not parse JSON from {cid}.")
                        parse_errors += 1

                    final_submission.append({'case_id': cid, 'prediction': prediction_data})
                else:
                    print(f"❌ No choices for {cid}.")
                    parse_errors += 1

            except Exception as e:
                print(f"❌ Error parsing {cid}: {e}")
                parse_errors += 1

        out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_file_name)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(final_submission, f, indent=2)
        print(f"✅ Saved {len(final_submission)} cases to {out_file}")
        if parse_errors:
            print(f"⚠️  {parse_errors} parse errors encountered.")
        if args.split == "dev":
            print("   Run eval_subtask4.py to evaluate results.")

if __name__ == "__main__":
    main()
