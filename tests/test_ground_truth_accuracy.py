#!/usr/bin/env python3
"""
Automated Ground Truth Accuracy Benchmark Suite for Medical OCR Engines.
Compares extraction outputs from 'native', 'hybrid', and 'vocr' engines
against canonical Ground Truth JSONs in pdf/ground_truth/.
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import Dict, Any, List, Tuple

# Color codes for clean CLI reporting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BENCHMARK_CASES = [
    {
        "name": "Drlogy UPCR Report",
        "file": os.path.join(BASE_DIR, "pdf/urine-protein-creatinine-ratio-test-report-format-example-sample-template-drlogy-lab-report.webp"),
        "ground_truth": os.path.join(BASE_DIR, "pdf/ground_truth/upcr.json")
    },
    {
        "name": "Full Blood Count (FBC)",
        "file": os.path.join(BASE_DIR, "pdf/FBC.pdf"),
        "ground_truth": os.path.join(BASE_DIR, "pdf/ground_truth/fbc.json")
    },
    {
        "name": "Hana Biochemistry",
        "file": os.path.join(BASE_DIR, "pdf/1.webp"),
        "ground_truth": os.path.join(BASE_DIR, "pdf/ground_truth/biochemistry_1.json")
    },
    {
        "name": "Renal Function / eGFR",
        "file": os.path.join(BASE_DIR, "pdf/EGFR.pdf"),
        "ground_truth": os.path.join(BASE_DIR, "pdf/ground_truth/egfr.json")
    },
    {
        "name": "Lipid Profile Panel",
        "file": os.path.join(BASE_DIR, "pdf/LIPID PROFILE.pdf"),
        "ground_truth": os.path.join(BASE_DIR, "pdf/ground_truth/lipid_profile.json")
    }
]

def clean_val(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower().replace(" ", "").replace(",", "")

def compare_demographics(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[int, int, List[str]]:
    """Compares patient demographics fields and returns (matched_fields, total_fields, notes)."""
    fields_to_check = ["patient_name", "pid_no", "age", "sex"]
    matched = 0
    total = len(fields_to_check)
    notes = []

    for field in fields_to_check:
        exp_val = clean_val(expected.get(field))
        act_val = clean_val(actual.get(field))

        if not exp_val:
            total -= 1
            continue

        if exp_val in act_val or act_val in exp_val:
            matched += 1
            notes.append(f"  ✓ {field}: '{actual.get(field)}'")
        else:
            notes.append(f"  ✗ {field}: Expected '{expected.get(field)}', Got '{actual.get(field)}'")

    return matched, max(total, 1), notes

def compare_results(actual_results: List[Dict[str, Any]], expected_results: List[Dict[str, Any]]) -> Tuple[int, int, List[str]]:
    """Compares observation items and returns (matched_items, total_expected, notes)."""
    matched = 0
    total = len(expected_results)
    notes = []

    act_dict = {clean_val(item.get("name")): item for item in actual_results}
    # Also index by type
    act_type_dict = {clean_val(item.get("type")): item for item in actual_results}

    for exp in expected_results:
        exp_name_clean = clean_val(exp.get("name"))
        exp_type_clean = clean_val(exp.get("type"))
        exp_val = clean_val(exp.get("value"))

        found_item = act_dict.get(exp_name_clean) or act_type_dict.get(exp_type_clean)
        if not found_item:
            # Fuzzy match
            for k, v in act_dict.items():
                if exp_type_clean in k or k in exp_type_clean:
                    found_item = v
                    break

        if found_item:
            act_val = clean_val(found_item.get("value"))
            val_match = (exp_val == act_val) or (exp_val in act_val) or (act_val in exp_val)
            if val_match:
                matched += 1
                notes.append(f"  ✓ {exp.get('name')}: {found_item.get('value')} {found_item.get('unit')}")
            else:
                notes.append(f"  ~ {exp.get('name')}: Val mismatch (Exp '{exp.get('value')}', Got '{found_item.get('value')}')")
        else:
            notes.append(f"  ✗ {exp.get('name')}: Not extracted")

    return matched, max(total, 1), notes

def run_benchmark(api_url: str, engine: str, model: str = None):
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN} 🧪 MEDICAL OCR ACCURACY BENCHMARK SUITE - GROUND TRUTH EVALUATION {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"📡 API Endpoint : {api_url}")
    print(f"⚙️ OCR Engine   : {BOLD}{engine.upper()}{RESET}")
    if model:
        print(f"🧠 Target Model : {model}")
    print(f"📁 Benchmark Set: {len(BENCHMARK_CASES)} canonical lab reports\n")

    total_demo_matched = 0
    total_demo_fields = 0
    total_res_matched = 0
    total_res_items = 0
    total_latency = 0.0

    scorecard_rows = []

    for i, case in enumerate(BENCHMARK_CASES, 1):
        print(f"{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"[{i}/{len(BENCHMARK_CASES)}] 📄 Testing: {BOLD}{case['name']}{RESET} ({case['file']})")

        if not os.path.exists(case["file"]):
            print(f"  {RED}File missing: {case['file']}{RESET}")
            continue

        if not os.path.exists(case["ground_truth"]):
            print(f"  {RED}Ground Truth JSON missing: {case['ground_truth']}{RESET}")
            continue

        with open(case["ground_truth"], "r") as gf:
            ground_truth = json.load(gf)

        # Upload document
        data_payload = {"engine": engine, "format": "json"}
        if model:
            data_payload["model"] = model

        t0 = time.monotonic()
        try:
            with open(case["file"], "rb") as f_doc:
                resp = requests.post(
                    f"{api_url}/api/ocr/upload",
                    files={"file": (os.path.basename(case["file"]), f_doc)},
                    data=data_payload,
                    timeout=120
                )
            duration = round(time.monotonic() - t0, 2)
            total_latency += duration

            if resp.status_code != 200:
                print(f"  {RED}API Request Failed with {resp.status_code}: {resp.text}{RESET}")
                scorecard_rows.append((case["name"], "FAILED", "0%", "0%", f"{duration}s"))
                continue

            resp_json = resp.json()
            extracted_data = resp_json.get("data", {})
            if isinstance(extracted_data, str):
                try:
                    extracted_data = json.loads(extracted_data)
                except Exception:
                    extracted_data = {}

            actual_patient = extracted_data.get("patient_info") or {}
            actual_results = extracted_data.get("results") or []

            # 1. Demographics Match
            d_matched, d_total, d_notes = compare_demographics(actual_patient, ground_truth.get("patient_info", {}))
            demo_pct = round((d_matched / d_total) * 100)
            total_demo_matched += d_matched
            total_demo_fields += d_total

            # 2. Results Match
            r_matched, r_total, r_notes = compare_results(actual_results, ground_truth.get("results", []))
            res_pct = round((r_matched / r_total) * 100)
            total_res_matched += r_matched
            total_res_items += r_total

            # Print details
            print(f"  ⏱️ Time Taken: {BOLD}{duration}s{RESET}")
            print(f"  👤 Demographics Accuracy: {GREEN if demo_pct >= 80 else YELLOW}{demo_pct}% ({d_matched}/{d_total}){RESET}")
            for n in d_notes:
                print(f"    {n}")

            print(f"  🧪 Observation Accuracy : {GREEN if res_pct >= 80 else YELLOW}{res_pct}% ({r_matched}/{r_total}){RESET}")
            for n in r_notes:
                print(f"    {n}")

            scorecard_rows.append((case["name"], "200 OK", f"{demo_pct}%", f"{res_pct}%", f"{duration}s"))

        except Exception as err:
            print(f"  {RED}Exception occurred: {err}{RESET}")
            scorecard_rows.append((case["name"], "ERROR", "0%", "0%", "N/A"))

    # Final Scorecard Summary
    overall_demo_pct = round((total_demo_matched / max(total_demo_fields, 1)) * 100)
    overall_res_pct = round((total_res_matched / max(total_res_items, 1)) * 100)
    avg_latency = round(total_latency / len(BENCHMARK_CASES), 2)

    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN} 🏆 BENCHMARK ACCURACY SCORECARD SUMMARY ({engine.upper()}) {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{'Report Name':<32} | {'Status':<8} | {'Demographics':<12} | {'Observations':<12} | {'Latency':<8}")
    print("-" * 80)
    for r_name, r_stat, r_dem, r_obs, r_lat in scorecard_rows:
        color = GREEN if "100%" in r_obs or "80%" in r_obs else YELLOW
        print(f"{r_name:<32} | {r_stat:<8} | {r_dem:<12} | {color}{r_obs:<12}{RESET} | {r_lat:<8}")
    print("-" * 80)
    print(f"{BOLD}🎯 OVERALL DEMOGRAPHICS ACCURACY : {GREEN if overall_demo_pct >= 80 else YELLOW}{overall_demo_pct}% ({total_demo_matched}/{total_demo_fields}){RESET}")
    print(f"{BOLD}🎯 OVERALL OBSERVATION ACCURACY  : {GREEN if overall_res_pct >= 80 else YELLOW}{overall_res_pct}% ({total_res_matched}/{total_res_items}){RESET}")
    print(f"{BOLD}⚡ AVERAGE PROCESSING LATENCY    : {avg_latency}s / report{RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Ground Truth Accuracy Benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8200", help="OCR API Base URL")
    parser.add_argument("--engine", type=str, default="native", choices=["native", "hybrid", "vocr"], help="OCR Pipeline Engine")
    parser.add_argument("--model", type=str, default=None, help="Target LLM model for hybrid/vocr")
    args = parser.parse_args()

    run_benchmark(api_url=args.url, engine=args.engine, model=args.model)
