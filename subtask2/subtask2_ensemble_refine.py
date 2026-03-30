import json
import os
import xml.etree.ElementTree as ET
import time
import requests
import shutil
import zipfile
from hybrid_subtask2 import parse_xml_case

# API Config
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
BASE_URL = "https://api.x.ai/v1"
MODEL_NAME = "grok-4-1-fast-reasoning"

HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Few-shot Examples (derived from Dev Set key)
FEW_SHOT_EXAMPLES = """
Example 1:
Question: Why was ERCP recommended over medication for CBD sludge?
Note:
[1] During the ERCP a pancreatic stent was required... and a common bile duct stent was placed to allow drainage of the biliary obstruction caused by stones and sludge.
[2] However, due to the patient's elevated INR, no sphincterotomy or stone removal was performed.
[3] Frank pus was noted to be draining from the common bile duct, and post-ERCP it was recommended that the patient remain on IV Zosyn for at least a week.
[4] The Vancomycin was discontinued.
[5] On hospital day 4 (post-procedure day 3) the patient returned to ERCP for re-evaluation of her biliary stent as her LFTs and bilirubin continued an upward trend.
[6] On ERCP the previous biliary stent was noted to be acutely obstructed by biliary sludge and stones.
[7] As the patient's INR was normalized to 1.2, a sphincterotomy was safely performed, with removal of several biliary stones in addition to the common bile duct stent.
Answer: {"relevant_ids": ["1", "5", "6", "7"]}

Example 2:
Question: Why was the patient given Lasix and oxygen reduced?
Note:
[1] Acute diastolic heart failure: Pt developed signs and symptoms of volume overload with shortness of breath, increased oxygen requirement and lower extremity edema.
[4] He was diuresed with lasix IV... then transitioned to PO torsemide with improvement in symptoms, although remained on a small amount of supplemental oxygen for comfort.
[5] Respiratory failure: The patient was intubated... and was given 8 L on his presentation to help maintain his BP's.
[6] This undoubtedly contributed to his continued hypoxemic respiratory failure.
Answer: {"relevant_ids": ["1", "4", "5", "6"]}
"""

# Full-Context CoT Prompt (Recall-Oriented)
FULL_CONTEXT_PROMPT = """You are an expert medical documentation specialist.
Your task is to identify the COMPREHENSIVE set of sentences from the clinical note that provide EVIDENCE to answer the clinician's question.

**Instructions:**
1. Read the **Clinician Question** and the **Clinical Note** carefully.
2. Select **ALL** sentences that:
    - Provide **direct answers** to the question.
    - Provide **necessary context** (e.g., timeline, initial conditions, reasons for changes).
    - Explain **WHY** a decision was made or **HOW** an outcome occurred.
    - Detail specific **values, dates, or test results** relevant to the inquiry.
3. **Completeness is critical.** Do not miss any supporting details. It is better to include a borderline sentence that provides context than to miss it.
4. Ensure you capture the **full narrative arc** (e.g., Problem -> Intervention -> Complication -> Resolution).

**Few-shot Examples:**
{examples}

**Current Task:**
**Clinician Question:** {question}

**Clinical Note:**
{note_text}

**Analyze step-by-step:**
1. Identify the core inquiry in the question.
2. Trace the relevant storyline in the note (chronological events, cause-and-effect).
3. Select ALL sentence IDs that form this complete evidence chain.

**Output Format:** Respond ONLY with valid JSON:
{{"relevant_ids": ["1", "3", "5"]}}"""

# Determine dataset path dynamically
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

def generate_full_context_data(mode="dev"):
    """Generate full context data for selected dataset (supporting multiple files)."""
    xml_paths = get_dataset_path(mode)
    all_cases_data = {}
    
    for xml_path in xml_paths:
        if not os.path.exists(xml_path):
            print(f"⚠️ Warning: Path not found: {xml_path}")
            continue
            
        print(f"📄 Loading data from: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for case_elem in root.findall('case'):
            case = parse_xml_case(case_elem)
            case_id = case['case_id']
            
            # Format full note with IDs
            full_note_text = ""
            # Sort IDs numerically to ensure correct order
            sorted_ids = sorted(case['sentences'].keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            for sent_id in sorted_ids:
                sentence_text = case['sentences'][sent_id]
                full_note_text += f"[{sent_id}] {sentence_text}\n"
            
            all_cases_data[case_id] = {
                'question': case['clinician_question'],
                'note_text': full_note_text,
                'sentence_ids': sorted_ids  # List of all valid IDs
            }
    
    return all_cases_data

def create_batch_requests(cases_data, num_samples=1):
    """Create batch requests. If num_samples > 1, create multiple requests per case."""
    batch_requests = []
    
    # Use higher temp for voting, 0.0 for single run
    temperature = 0.7 if num_samples > 1 else 0.0
    
    for case_id, data in cases_data.items():
        question = data['question']
        note_text = data['note_text']
        
        # Construct Prompt
        prompt = FULL_CONTEXT_PROMPT.format(
            examples=FEW_SHOT_EXAMPLES,
            question=question,
            note_text=note_text.strip()
        )
        
        for i in range(num_samples):
            # Request ID format: "caseid" or "caseid_sample_0"
            req_id = f"{case_id}" if num_samples == 1 else f"{case_id}_sample_{i}"
            
            request = {
                "batch_request_id": req_id,
                "batch_request": {
                    "chat_get_completion": {
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": "You are a precise medical evidence classifier. Respond ONLY with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": 1000
                    }
                }
            }
            batch_requests.append(request)
    
    return batch_requests

def upload_batch(batch_requests, batch_name="subtask2_full_context"):
    """Upload batch to xAI API."""
    print(f"\n📦 Creating batch '{batch_name}' with {len(batch_requests)} requests...")
    
    try:
        print(f"   Requesting batch creation for '{batch_name}'...")
        batch_response = requests.post(
            f"{BASE_URL}/batches",
            headers=HEADERS,
            json={"name": batch_name},
            timeout=30
        )
        
        if batch_response.status_code != 200:
            print(f"❌ Batch creation failed: {batch_response.text}")
            return None
        
        batch_info = batch_response.json()
        batch_id = batch_info.get('batch_id', batch_info.get('id'))
        
        print(f"✅ Batch created! ID: {batch_id}")
        
        # Upload requests in chunks
        chunk_size = 50 # Reduced from 50 to avoid timeouts with large full-context payloads
        total_uploaded = 0
        
        for i in range(0, len(batch_requests), chunk_size):
            chunk = batch_requests[i:i+chunk_size]
            current_batch_num = (i // chunk_size) + 1
            total_batches = (len(batch_requests) + chunk_size - 1) // chunk_size
            
            print(f"   Uploading chunk {current_batch_num}/{total_batches} ({len(chunk)} reqs)...", end=' ')
            
            try:
                resp = requests.post(
                    f"{BASE_URL}/batches/{batch_id}/requests",
                    headers=HEADERS,
                    json={"batch_requests": chunk},
                    timeout=120 # Long timeout for large payloads
                )
                
                if resp.status_code == 200:
                    total_uploaded += len(chunk)
                    print(f"✅ OK")
                else:
                    print(f"❌ Failed ({resp.status_code}): {resp.text}")
            except requests.exceptions.Timeout:
                print(f"❌ Timeout! (Chunk too large?)")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            time.sleep(0.5)
        
        print(f"\n🎉 Upload complete! Total: {total_uploaded} requests")
        print(f"   Batch ID: {batch_id}")
        
        return batch_id
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def check_batch_status(batch_id):
    """Check batch status."""
    response = requests.get(
        f"{BASE_URL}/batches/{batch_id}",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        info = response.json()
        print("\n--- Batch Status ---")
        print(f"ID: {info['id']}")
        print(f"Status: {info['status']}")
        print(f"Request Counts: {info.get('request_counts', {})}")
        return info['status']
    else:
        print(f"❌ Error: {response.text}")
        return None

def download_results(batch_id, cases_data):
    """Download and process batch results."""
    print(f"\nFetching results for batch {batch_id}...")
    
    all_results = []
    pagination_token = None
    
    while True:
        url = f"{BASE_URL}/batches/{batch_id}/results?page_size=100"
        if pagination_token:
            url += f"&pagination_token={pagination_token}"
        
        res_resp = requests.get(url, headers=HEADERS)
        
        if res_resp.status_code != 200:
            print(f"❌ Error fetching results: {res_resp.text}")
            break
        
        data = res_resp.json()
        all_results.extend(data.get('results', []))
        
        pagination_token = data.get('pagination_token')
        if not pagination_token:
            break
    
    print(f"✅ Fetched {len(all_results)} results")
    
    # Parse results
    predictions = {} 
    
    for result in all_results:
        request_id = result.get('batch_request_id') or result.get('custom_id') or ''
        
        if not request_id:
            continue
        
        # Determine base Case ID (handle _sample_ suffix)
        if "_sample_" in request_id:
            case_id = request_id.split("_sample_")[0]
        else:
            case_id = request_id
            
        if case_id not in cases_data:
            continue
            
        # Get response
        batch_result = result.get('batch_result', {})
        response = batch_result.get('response', {})
        
        # Target node containing 'choices'
        target_node = response.get('chat_get_completion', {})
        if not target_node:
             target_node = response.get('body', {})
             
        choices = target_node.get('choices', [])
        
        if not choices:
            continue
        
        content = choices[0].get('message', {}).get('content', '').strip()
        
        # Parse JSON - handle cases where LLM adds explanation after JSON
        try:
            # Remove markdown fences if present
            clean_content = content.replace('```json', '').replace('```', '').strip()
            
            # Simple brace matching to isolate JSON object
            start_idx = clean_content.find('{')
            if start_idx == -1:
                # No braces found? Force fallback
                raise ValueError("No JSON start brace found")
            
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(clean_content)):
                if clean_content[i] == '{':
                    brace_count += 1
                elif clean_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                 # Incomplete brace match? Use what we have or fallback
                 raise ValueError("Incomplete JSON object")
            
            json_part = clean_content[start_idx:end_idx]
            
            parsed = json.loads(json_part)
            relevant_ids = parsed.get('relevant_ids', [])
            
            valid_ids = cases_data[case_id]['sentence_ids']
            predictions[request_id] = [str(sid) for sid in relevant_ids if str(sid) in valid_ids]
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback 1: Regex Search for relevant_ids pattern
            import re
            match = re.search(r'"relevant_ids"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            found_ids = []
            
            if match:
                ids_str = match.group(1)
                # Find all quoted numbers
                found_ids = re.findall(r'"(\d+)"', ids_str)
                # If no quoted numbers, try unquoted
                if not found_ids:
                     found_ids = re.findall(r'(\d+)', ids_str)
                
                if found_ids:
                     # print(f"⚠️  Regex Rescue for Request {request_id}: Found {len(found_ids)} IDs")
                     pass

            # Fallback 2: If regex specific pattern failed, try finding all quoted numbers in content (Last Resort)
            if not found_ids:
                 found_ids = re.findall(r'"(\d+)"', content)
                 if found_ids:
                     # print(f"⚠️  Broad Regex Fallback for Request {request_id}: Found {len(found_ids)} IDs")
                     pass
            
            if found_ids:
                valid_ids = cases_data[case_id]['sentence_ids']
                predictions[request_id] = [sid for sid in found_ids if sid in valid_ids]
            else:
                print(f"\n❌ Parse Error Request {request_id}: {str(e)}")
                # print(f"   Content: {content[:100]}...")
    
    return predictions

def aggregate_votes_manual(predictions, threshold):
    """Aggregate predictions using Majority Voting with custom threshold."""
    print(f"\n🗳️ Aggregating votes (Threshold >= {threshold})...")
    final_preds = {}
    
    # Group by base case_id
    case_votes = {} # {case_id: {sent_id: count}}
    
    for req_id, sent_ids in predictions.items():
        if "_sample_" in req_id:
            base_id = req_id.split("_sample_")[0]
        else:
            base_id = req_id
            
        if base_id not in case_votes:
            case_votes[base_id] = {}
        
        # Count occurences of each sentence ID
        for sid in set(sent_ids): # Count once per sample
            case_votes[base_id][sid] = case_votes[base_id].get(sid, 0) + 1
            
    for case_id, votes in case_votes.items():
        selected = [sid for sid, count in votes.items() if count >= threshold]
        # Sort numerically
        selected.sort(key=lambda x: int(x) if x.isdigit() else x)
        final_preds[case_id] = selected
        
    print(f"✅ Aggregated {len(final_preds)} cases.")
    return final_preds

def package_submission(output_file, mode="dev"):
    """Package results into submission.zip inside subtask/mode folder."""
    # Define folder structure: subtask_2/test/ or subtask_2/dev/
    subtask_dir = os.path.join("subtask_2", mode)
    
    if os.path.exists(subtask_dir):
        shutil.rmtree(subtask_dir)
    os.makedirs(subtask_dir)
    
    # Copy JSON to folder with standard name 'submission.json'
    target_json = os.path.join(subtask_dir, "submission.json")
    shutil.copy(output_file, target_json)
    
    # Create ZIP inside the folder: subtask_2/test/submission.zip
    zip_path = os.path.join(subtask_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(target_json, arcname="submission.json")
    
    print(f"📦 Created submission package inside '{subtask_dir}':")
    print(f"   - {target_json}")
    print(f"   - {zip_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Subtask 2: Full-Context CoT Evidence Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subtask2_ensemble_refine.py upload --mode test\n"
            "  python subtask2_ensemble_refine.py upload --mode test --samples 5\n"
            "  python subtask2_ensemble_refine.py status --batch-id <BATCH_ID>\n"
            "  python subtask2_ensemble_refine.py download --batch-id <BATCH_ID> --mode test\n"
            "  python subtask2_ensemble_refine.py download --batch-id <BATCH_ID> --mode test --voting\n"
        )
    )
    parser.add_argument("action", choices=["upload", "status", "download"],
                        help="Action to perform")
    parser.add_argument("--mode", choices=["dev", "test"], default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--samples", type=int, choices=[1, 5], default=1,
                        help="Samples per case: 1=standard, 5=self-consistency (default: 1)")
    parser.add_argument("--batch-id", dest="batch_id",
                        help="Batch ID (required for status and download)")
    parser.add_argument("--voting", action="store_true",
                        help="Apply majority voting when downloading (for --samples 5 runs)")
    parser.add_argument("--threshold", type=int,
                        help="Voting threshold (default: majority+1); only used with --voting")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("Subtask 2: Full-Context CoT Extraction (No BM25)")
    print("="*50)
    print(f"Mode: {args.mode.upper()}")

    if args.action == "upload":
        cases_data = generate_full_context_data(mode=args.mode)

        c_file = os.path.join(PROJECT_ROOT, f"full_data_{args.mode}.json")
        with open(c_file, 'w', encoding='utf-8') as f:
            json.dump(cases_data, f, indent=2)

        batch_requests = create_batch_requests(cases_data, num_samples=args.samples)
        print(f"📊 {len(batch_requests)} requests prepared ({args.samples} per case)")

        batch_name = f"subtask2_{args.mode}_n{args.samples}_{int(time.time())}"
        batch_id = upload_batch(batch_requests, batch_name)

        if batch_id:
            print(f"\n✅ Batch uploaded successfully!")
            print(f"   Save this ID: {batch_id}")

    elif args.action == "status":
        if not args.batch_id:
            parser.error("--batch-id is required for the status action")
        check_batch_status(args.batch_id)

    elif args.action == "download":
        if not args.batch_id:
            parser.error("--batch-id is required for the download action")

        c_file = os.path.join(PROJECT_ROOT, f"full_data_{args.mode}.json")
        if not os.path.exists(c_file):
            print(f"❌ Data file not found: {c_file}")
            print(f"   Run: python subtask2_ensemble_refine.py upload --mode {args.mode}")
            return

        with open(c_file, 'r', encoding='utf-8') as f:
            cases_data = json.load(f)

        predictions = download_results(args.batch_id, cases_data)
        final_predictions = predictions

        if args.voting:
            sample_counts = {}
            for req_id in predictions.keys():
                base = req_id.split("_sample_")[0]
                sample_counts[base] = sample_counts.get(base, 0) + 1
            est_samples = max(sample_counts.values()) if sample_counts else 1
            threshold = args.threshold if args.threshold else (est_samples // 2 + 1)
            print(f"   Majority voting: threshold {threshold}/{est_samples}")
            final_predictions = aggregate_votes_manual(predictions, threshold)

        submission = [
            {'case_id': cid, 'prediction': preds}
            for cid, preds in final_predictions.items()
        ]

        output_file = os.path.join(PROJECT_ROOT, f"submission_subtask2_{args.mode}_full.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(submission, f, indent=2)

        print(f"\n✅ Saved to: {output_file}")
        package_submission(output_file, args.mode)

        if args.mode == 'dev':
            print("   Run eval_subtask2.py to evaluate results.")

if __name__ == "__main__":
    main()
