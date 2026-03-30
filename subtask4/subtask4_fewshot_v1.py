"""
Subtask 4: Evidence Alignment
Aligns each answer sentence to supporting sentences in the clinical note excerpt.
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
DEV_XML = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "dev", "archehr-qa.xml")
DEV_KEY = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "dev", "archehr-qa_key.json")
TEST1_XML = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test", "archehr-qa.xml")
TEST1_KEY = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test", "archehr-qa_key.json")
TEST2_XML = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test-2026", "archehr-qa.xml")
TEST2_KEY = os.path.join(PROJECT_ROOT, "data", "subtask4", "v1.5", "test-2026", "archehr-qa_key.json")

HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are an expert clinical evidence aligner.
Your task is to align each answer sentence with the supporting sentence(s) from the clinical note excerpt.
For each answer sentence (identified by Answer ID), return a JSON object containing the Answer ID and an array of Evidence IDs (from the Note Sentences) that support it.
If an answer sentence is not supported by any note sentence, return an empty array [] for its evidence_id.

Constraints:
1. Return ONLY valid JSON as a list of objects exactly matching the requested format. DO NOT WRAP in Markdown blocks.
2. Format: [{"answer_id": "1", "evidence_id": ["2", "5"]}, {"answer_id": "2", "evidence_id": []}]
3. Avoid over-citing. Only select note sentences that directly and explicitly support the answer.
4. Alignments are many-to-many.
"""

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
            
        data[cid] = {'clinician_question': clin_q, 'patient_question': pat_q, 'note_sentences': sents}
    return data

def get_few_shot_examples():
    """Load good examples from Dev set."""
    if not os.path.exists(DEV_KEY): return ""
    
    with open(DEV_KEY, 'r', encoding='utf-8') as f:
        gold = json.load(f)
        
    xml_data = parse_xml_sentences(DEV_XML)
    
    examples = ""
    # Use first 2 cases as examples to save context length
    for item in gold[:2]:
        cid = item['case_id']
        clin_q = xml_data[cid]['clinician_question']
        
        note_text = ""
        for sid, text in sorted(xml_data[cid]['note_sentences'].items(), key=lambda x: int(x[0])):
            note_text += f"[{sid}] {text}\n"
            
        answer_text = ""
        expected_output = []
        for ans in item['clinician_answer_sentences']:
            ans_id = ans.get('id')
            ans_text = ans.get('text')
            answer_text += f"[Answer {ans_id}] {ans_text}\n"
            citations = ans.get('citations', '')
            if citations:
                ev_ids = [c.strip() for c in citations.split(',')]
            else:
                ev_ids = []
            expected_output.append({"answer_id": ans_id, "evidence_id": ev_ids})
            
        examples += f"**Example Case:**\n**Clinician Question:** {clin_q}\n**Note Excerpt:**\n{note_text}\n**Answer Sentences:**\n{answer_text}\n**Output:** {json.dumps(expected_output)}\n\n"
        
    return examples

def create_batch_requests(test_key_data, test_xml_data, examples):
    batch_requests = []
    
    for item in test_key_data:
        case_id = item['case_id']
        info = test_xml_data[case_id]
        
        note_text = ""
        for sid, text in sorted(info['note_sentences'].items(), key=lambda x: int(x[0])):
            note_text += f"[{sid}] {text}\n"
            
        answer_text = ""
        for ans in item['clinician_answer_sentences']:
            ans_id = ans.get('id')
            ans_text = ans.get('text')
            answer_text += f"[Answer {ans_id}] {ans_text}\n"
            
        user_content = f"{examples}**Current Case:**\n**Clinician Question:** {info['clinician_question']}\n**Note Excerpt:**\n{note_text}\n**Answer Sentences:**\n{answer_text}\n**Output:**"
        
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

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Subtask 4 v1: Evidence Alignment (Few-Shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subtask4_fewshot_v1.py upload --split test\n"
            "  python subtask4_fewshot_v1.py status --batch-id <BATCH_ID>\n"
            "  python subtask4_fewshot_v1.py download --batch-id <BATCH_ID> --split test\n"
        )
    )
    parser.add_argument("action", choices=["upload", "status", "download"],
                        help="Action to perform")
    parser.add_argument("--split", choices=["dev", "test"], default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--batch-id", dest="batch_id",
                        help="Batch ID (required for status and download)")
    args = parser.parse_args()

    print("\n=== Subtask 4: Evidence Alignment ===\n")

    # Load data
    print("📖 Loading Data...")
    if not os.path.exists(DEV_XML) or not os.path.exists(DEV_KEY):
        print("❌ Dev data files missing.")
        dev_key_data, dev_xml_data = [], {}
    else:
        dev_xml_data = parse_xml_sentences(DEV_XML)
        with open(DEV_KEY, 'r', encoding='utf-8') as f:
            dev_key_data = json.load(f)

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

    if args.action == "upload":
        if args.split == "dev":
            batch_reqs = create_batch_requests(dev_key_data, dev_xml_data, examples)
            bname = f"subtask4_dev_{int(time.time())}"
        else:
            batch_reqs = create_batch_requests(test_key_data, test_xml_data, examples)
            bname = f"subtask4_test_{int(time.time())}"

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

                    import re
                    prediction_data = []
                    if "```json" in content:
                        clean_content = content.split("```json")[-1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[-1].split("```")[0].strip()
                    else:
                        clean_content = content

                    parsed = False
                    try:
                        prediction_data = json.loads(clean_content)
                        parsed = True
                    except Exception:
                        for closure in ["]", "}]", '"}]']:
                            try:
                                prediction_data = json.loads(clean_content + closure)
                                parsed = True
                                break
                            except Exception:
                                pass

                    if not parsed:
                        matches = re.findall(r'\{[^{}]*"answer_id"[^{}]*"evidence_id"[^{}]*\}', content)
                        for m in matches:
                            try:
                                prediction_data.append(json.loads(m))
                            except Exception:
                                pass

                    if not prediction_data:
                        print(f"❌ Could not parse JSON from {cid}.")

                    final_submission.append({'case_id': cid, 'prediction': prediction_data})
                else:
                    print(f"❌ No choices for {cid}.")

            except Exception as e:
                print(f"❌ Error parsing {cid}: {e}")

        out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_file_name)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(final_submission, f, indent=2)
        print(f"✅ Saved {len(final_submission)} cases to {out_file}")
        if args.split == "dev":
            print("   Run eval_subtask4.py to evaluate results.")

if __name__ == "__main__":
    main()
