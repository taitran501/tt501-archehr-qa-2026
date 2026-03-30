"""
Subtask 3: Answer Generation
Generates natural language answers based on retrieved evidence (from Subtask 2).
Strictly follows constraints: < 75 words, Professional Tone, Third-Person.
"""

import json
import os
import time
import requests
import xml.etree.ElementTree as ET

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
BASE_URL = "https://api.x.ai/v1"
MODEL_NAME = "grok-4-1-fast-reasoning"

# Paths
DEV_XML = os.path.join(PROJECT_ROOT, "data", "v1.4", "dev", "archehr-qa.xml")
DEV_KEY = os.path.join(PROJECT_ROOT, "data", "v1.4", "dev", "archehr-qa_key.json")
TEST_XML = os.path.join(PROJECT_ROOT, "data", "v1.4", "test-2026", "archehr-qa.xml")

# Input Evidence File (Result from Subtask 2 — output of subtask2_ensemble_refine.py in test mode)
EVIDENCE_FILE = os.path.join(PROJECT_ROOT, "submission_subtask2_test_full.json")

HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are a professional medical scribe.
Your task is to draft a concise clinical answer to a clinician's question, based ONLY on the provided evidence sentences.

**Constraints:**
1. **Length:** Maximum 75 words (approx. 5 sentences).
2. **Tone:** Professional, detached, medical register (e.g., "The patient...", "Mr. X..."). Do NOT use "You" or conversational language.
3. **Grounding:** Use ONLY the provided evidence. Do not hallucinate or use external knowledge.
4. **Format:** Plain text paragraph. No citations. No markdown.
"""

def parse_xml_sentences(xml_path):
    """Parse XML to get {case_id: {sent_id: text, question: text}}"""
    tree = ET.parse(xml_path)
    data = {}
    for case in tree.getroot().findall('case'):
        cid = case.get('id')
        clin_q = case.find('clinician_question').text.strip()
        
        sents = {}
        for s in case.findall('.//note_excerpt_sentences/sentence'):
            sents[s.get('id')] = s.text.strip()
            
        data[cid] = {'question': clin_q, 'sentences': sents}
    return data

def get_few_shot_examples():
    """Load good examples from Dev set."""
    if not os.path.exists(DEV_KEY): return ""
    
    with open(DEV_KEY, 'r') as f:
        gold = json.load(f)
        
    xml_data = parse_xml_sentences(DEV_XML)
    
    examples = ""
    # Use first 3 cases as examples
    for item in gold[:3]:
        cid = item['case_id']
        question = xml_data[cid]['question']
        answer = item['clinician_answer_without_citations']
        
        # Get evidence texts
        # In gold, specific sentences are cited, but for strict grounding, 
        # let's assume we provide the sentences that were marked 'essential'
        evidence_ids = [a['sentence_id'] for a in item['answers'] if a['relevance'] == 'essential']
        
        evidence_text = ""
        for sid in evidence_ids:
            txt = xml_data[cid]['sentences'].get(sid, "")
            evidence_text += f"[{sid}] {txt}\n"
            
        examples += f"**Example Case:**\n**Question:** {question}\n**Evidence:**\n{evidence_text}\n**Answer:** {answer}\n\n"
        
    return examples

def create_batch_requests(test_data, evidence_map, examples):
    batch_requests = []
    
    for case_id, info in test_data.items():
        if case_id not in evidence_map:
            print(f"⚠️ Warning: No evidence found for Case {case_id}")
            continue
            
        pred_ids = evidence_map[case_id]
        
        # Construct evidence text
        evidence_text = ""
        # Sort IDs numerically
        sorted_ids = sorted(pred_ids, key=lambda x: int(x) if x.isdigit() else x)
        
        for sid in sorted_ids:
            if sid in info['sentences']:
                evidence_text += f"[{sid}] {info['sentences'][sid]}\n"
        
        user_content = f"{examples}**Current Case:**\n**Question:** {info['question']}\n**Evidence:**\n{evidence_text}\n**Answer:**"
        
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
                    "max_tokens": 150
                }
            }
        }
        batch_requests.append(request)
        
    return batch_requests

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Subtask 3: Answer Generation (RAG + Few-Shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subtask3_rag_fewshot.py upload\n"
            "  python subtask3_rag_fewshot.py status --batch-id <BATCH_ID>\n"
            "  python subtask3_rag_fewshot.py download --batch-id <BATCH_ID>\n"
            "\n"
            "Note: Run subtask2_ensemble_refine.py download --mode test first to\n"
            f"      generate the required evidence file: {os.path.basename(EVIDENCE_FILE)}\n"
        )
    )
    parser.add_argument("action", choices=["upload", "status", "download"],
                        help="Action to perform")
    parser.add_argument("--batch-id", dest="batch_id",
                        help="Batch ID (required for status and download)")
    args = parser.parse_args()

    print("\n=== Subtask 3: Answer Generation ===\n")

    # Check evidence file for upload and download
    if args.action in ("upload", "download") and not os.path.exists(EVIDENCE_FILE):
        print(f"❌ Evidence file not found: {EVIDENCE_FILE}")
        print("   Run: python subtask2_ensemble_refine.py download --mode test")
        return

    print("📖 Loading Data...")
    test_data = parse_xml_sentences(TEST_XML)
    examples = get_few_shot_examples()

    with open(EVIDENCE_FILE, 'r') as f:
        evidence_list = json.load(f)
        evidence_map = {item['case_id']: item['prediction'] for item in evidence_list}

    print(f"✅ Loaded {len(test_data)} test cases, evidence for {len(evidence_map)} cases.")

    if args.action == "upload":
        batch_reqs = create_batch_requests(test_data, evidence_map, examples)
        print(f"📦 Prepared {len(batch_reqs)} requests.")

        try:
            b_resp = requests.post(
                f"{BASE_URL}/batches",
                headers=HEADERS,
                json={"name": f"subtask3_test_{int(time.time())}"},
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
                    content = choices[0]['message']['content']
                    final_submission.append({'case_id': cid, 'prediction': content.strip()})
                else:
                    print(f"❌ No choices for {cid}. Keys: {list(body.keys())}")
                    if not final_submission:
                        print(f"DEBUG: {json.dumps(body, indent=2)}")
            except Exception as e:
                print(f"❌ Error parsing {cid}: {e}")

        out_file = os.path.join(PROJECT_ROOT, "submission_subtask3_test.json")
        with open(out_file, 'w') as f:
            json.dump(final_submission, f, indent=2)
        print(f"✅ Saved {len(final_submission)} cases to {out_file}")

if __name__ == "__main__":
    main()
