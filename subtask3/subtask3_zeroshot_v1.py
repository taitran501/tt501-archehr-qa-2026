"""
Subtask 3, Run 1: Answer Generation (Zero-Shot)
Uses the full note excerpt directly. No pre-selected evidence, no few-shot examples.
Model: grok-4-fast-reasoning
Output: JSON {"answer": "..."}, parsed to plain text for submission.
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
MODEL_NAME = "grok-4-fast-reasoning"

# Paths
TEST_XML = os.path.join(PROJECT_ROOT, "data", "v1.4", "test-2026", "archehr-qa.xml")

HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are a clinical documentation specialist extracting factual answers from medical notes.

Critical Guidelines:
1. Extract ONLY information explicitly stated in the note. No speculation, inference, or hedging.
2. Use professional medical register ("The patient..."). No empathy phrases.
3. 75 words maximum (strictly enforced).
4. Use past tense for completed medical events.
5. State facts directly without qualifiers."""


def parse_xml_cases(xml_path):
    """Parse XML to get {case_id: {patient_question, clinician_question, full_note_text}}."""
    tree = ET.parse(xml_path)
    data = {}
    for case in tree.getroot().findall("case"):
        cid = case.get("id")
        clin_q = case.find("clinician_question").text.strip()
        pat_q_elem = case.find("patient_narrative")
        pat_q = pat_q_elem.text.strip() if pat_q_elem is not None and pat_q_elem.text else ""

        sents = {}
        for s in case.findall(".//note_excerpt_sentences/sentence"):
            sents[s.get("id")] = s.text.strip()

        sorted_ids = sorted(sents.keys(), key=lambda x: int(x) if x.isdigit() else x)
        full_note = ""
        for sid in sorted_ids:
            full_note += f"[{sid}] {sents[sid]}\n"

        data[cid] = {
            "patient_question": pat_q,
            "clinician_question": clin_q,
            "full_note_text": full_note,
        }
    return data


def create_batch_requests(test_data):
    batch_requests = []
    for case_id, info in test_data.items():
        user_content = (
            f"Patient Question: {info['patient_question']}\n"
            f"Clinician Question: {info['clinician_question']}\n"
            f"Note Excerpt:\n{info['full_note_text']}\n"
            'Output Format: Respond ONLY with valid JSON:\n'
            '{"answer": "Your factual extraction here (max 75 words)"}'
        )
        request = {
            "batch_request_id": case_id,
            "batch_request": {
                "chat_get_completion": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200,
                }
            },
        }
        batch_requests.append(request)
    return batch_requests


def parse_answer(content):
    """Extract answer string from JSON response, with plain-text fallback."""
    import re

    # Clean markdown wrappers
    clean = content.strip()
    if "```json" in clean:
        clean = clean.split("```json")[-1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"].strip()
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback: extract value after "answer":
    match = re.search(r'"answer"\s*:\s*"(.*?)"', content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return content as-is (already plain text)
    return content.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Subtask 3, Run 1: Answer Generation (Zero-Shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subtask3_zeroshot_v1.py upload\n"
            "  python subtask3_zeroshot_v1.py status --batch-id <BATCH_ID>\n"
            "  python subtask3_zeroshot_v1.py download --batch-id <BATCH_ID>\n"
        ),
    )
    parser.add_argument("action", choices=["upload", "status", "download"],
                        help="Action to perform")
    parser.add_argument("--batch-id", dest="batch_id",
                        help="Batch ID (required for status and download)")
    args = parser.parse_args()

    print("\n=== Subtask 3, Run 1: Answer Generation (Zero-Shot) ===\n")

    if args.action == "upload":
        if not os.path.exists(TEST_XML):
            print(f"Test XML not found: {TEST_XML}")
            return

        test_data = parse_xml_cases(TEST_XML)
        print(f"Loaded {len(test_data)} test cases.")

        batch_reqs = create_batch_requests(test_data)
        print(f"Prepared {len(batch_reqs)} requests.")

        try:
            b_resp = requests.post(
                f"{BASE_URL}/batches",
                headers=HEADERS,
                json={"name": f"subtask3_zeroshot_test_{int(time.time())}"},
                timeout=30,
            )
            if b_resp.status_code != 200:
                print(f"Error creating batch: {b_resp.text}")
                return

            batch_info = b_resp.json()
            batch_id = batch_info.get("batch_id", batch_info.get("id"))
            print(f"Batch {batch_id} created. Uploading requests...")

            chunk_size = 50
            for i in range(0, len(batch_reqs), chunk_size):
                chunk = batch_reqs[i : i + chunk_size]
                requests.post(
                    f"{BASE_URL}/batches/{batch_id}/requests",
                    headers=HEADERS,
                    json={"batch_requests": chunk},
                    timeout=60,
                )

            print(f"Upload complete. Batch ID: {batch_id}")

        except Exception as e:
            print(f"Error: {e}")

    elif args.action == "status":
        if not args.batch_id:
            parser.error("--batch-id is required for the status action")
        resp = requests.get(f"{BASE_URL}/batches/{args.batch_id}", headers=HEADERS)
        if resp.status_code == 200:
            info = resp.json()
            print(f"Status: {info.get('status')}")
            counts = info.get("request_counts", {})
            if counts:
                print(f"Counts: {counts}")
        else:
            print(f"Error: {resp.text}")

    elif args.action == "download":
        if not args.batch_id:
            parser.error("--batch-id is required for the download action")

        print("Fetching results...")
        results = []
        url = f"{BASE_URL}/batches/{args.batch_id}/results?page_size=100"
        while url:
            resp = requests.get(url, headers=HEADERS)
            if resp.status_code != 200:
                print(f"Error fetching page: {resp.text}")
                break
            data = resp.json()
            results.extend(data.get("results", []))
            token = data.get("pagination_token")
            url = (
                f"{BASE_URL}/batches/{args.batch_id}/results?page_size=100&pagination_token={token}"
                if token
                else None
            )

        print(f"Fetched {len(results)} results.")

        final_submission = []
        for res in results:
            cid = res.get("batch_request_id") or res.get("custom_id")
            if not cid:
                continue
            try:
                batch_res = res.get("batch_result", {})
                response = batch_res.get("response", {})
                body = response.get("body", response)
                choices = body.get("choices", [])
                if not choices:
                    choices = body.get("chat_get_completion", {}).get("choices", [])
                if choices:
                    raw = choices[0]["message"]["content"]
                    answer = parse_answer(raw)
                    final_submission.append({"case_id": cid, "prediction": answer})
                else:
                    print(f"No choices for case {cid}.")
            except Exception as e:
                print(f"Error parsing case {cid}: {e}")

        out_file = os.path.join(PROJECT_ROOT, "submission_subtask3_zeroshot_test.json")
        with open(out_file, "w") as f:
            json.dump(final_submission, f, indent=2)
        print(f"Saved {len(final_submission)} cases to {out_file}")


if __name__ == "__main__":
    main()
