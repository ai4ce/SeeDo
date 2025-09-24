from __future__ import annotations
import os, json, argparse
from typing import Dict, Any
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("YOUR_API_KEY"))
MODEL  = "gpt-4o-2024-11-20"

def build_eval_prompt(gt_text: str, pr_text: str) -> str:
    """
    Generate an evaluation prompt that enforces strict JSON output.
    The model must output **one** JSON object only, with fixed fields and fixed types, so it can be written back to a CSV easily.
    """
    return f"""
You are an evaluation engine for action-sequence predictions.
Your ONLY output should be ONE JSON object, with EXACT keys and types:

{{
  "TSR": 0 or 1,
  "SSR": number in [0,1] or null,
  "FSR": 0 or 1,
  "Vision Error": 0 or 1,
  "Spatial Error": 0 or 1,
  "Temporal Error": 0 or 1
}}

### Key Ideas
- GT and PR each contain a list of actions.
- Normalize each action to (object, relation, reference).
- Relations: {{in, left_of, right_of, above, below, on_top_of}}
- "drop A and B in C" → separate steps: (A, in, C), (B, in, C)
- Special case: if A1 is already in B, then "drop A2 on A1" ≈ "drop A2 in B".

### Metrics
- **TSR** = 1 if GT and PR have same length and every step matches, else 0.
- **SSR** = LCS_length(PR, GT) / |GT| (use step-equivalence, null if |GT|=0).
- **FSR** = 1 if final positions of all objects match, else 0.

### Error Flags
- Compare GT and PR in order, find the FIRST mismatch:
  - If manipulated object differs → Vision Error = 1
  - Else if relation differs → Spatial Error = 1
  - Else if reference differs → Spatial Error = 1
  - Else if order or length mismatch → Temporal Error = 1
  - Only ONE error flag can be 1.

### Step Equivalence
Two steps are equivalent if:
- Object names are **visually similar** (very loose matching):
  - Any garments or clothes that **look similar** are equal.
    Examples: "blue shirt" ~ "navy hoodie", "plaid cloth" ~ "checkered garment".
  - Any vegetables/fruits/containers that **look similar** are equal.
    Examples: "red pepper" ~ "chili", "eggplant" ~ "purple garlic",
              "garbage can" ~ "bin", "glass container" ~ "container".
- Wooden blocks must match by exact ID.
  Example: wooden_block(ID:1) ≠ wooden_block(ID:2), this is a strict requirement and you must pay attention and follow!
- Relations must match exactly after normalization.
- References are judged with same loose visual rules as objects.
- For bowls, if the ground truth specifically distinguishes with ID, the answer should also distinguish it. But it can be with different color of contours or ID.

GT:
<<<
{gt_text}
>>>

PR:
<<<
{pr_text}
>>>

Return ONLY the JSON object, no extra text.
"""  # end f-string

# ========== Invoke the LLM to perform the evaluation. ==========
def evaluate_with_llm(gt_text: str, pr_text: str) -> Dict[str, Any]:
    prompt = build_eval_prompt(gt_text, pr_text)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON for the user request."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Parse the JSON and apply basic error handling.
    data = json.loads(raw)
    # Validate required fields
    for k in ["TSR","SSR","FSR","Vision Error","Spatial Error","Temporal Error"]:
        if k not in data:
            raise ValueError(f"LLM output missing key: {k}")
    # Normalize numeric types
    def _to_int01(v):
        return int(v) if v is not None else 0
    out = {
        "TSR": _to_int01(data["TSR"]),
        "SSR": (float(data["SSR"]) if data["SSR"] is not None else float("nan")),
        "FSR": _to_int01(data["FSR"]),
        "Vision Error": _to_int01(data["Vision Error"]),
        "Spatial Error": _to_int01(data["Spatial Error"]),
        "Temporal Error": _to_int01(data["Temporal Error"]),
    }
    # Consistency constraint (only one error bit allowed to be 1)
    err_sum = out["Vision Error"] + out["Spatial Error"] + out["Temporal Error"]
    if err_sum > 1:
        # If the model unexpectedly returns multiple error bits, keep the first non-zero and set the others to zero
        if out["Vision Error"]:
            out["Spatial Error"] = out["Temporal Error"] = 0
        elif out["Spatial Error"]:
            out["Temporal Error"] = 0
        else:
            out["Vision Error"] = out["Spatial Error"] = 0
    return out


# ========== Batch CSV processing ==========
def batch_eval_csv(gt_csv: str, ans_csv: str, out_csv: str) -> None:
    """
    Read the Ground Truth and Answer CSVs, align by demo
    use the LLM to evaluate and append six result columns, then write to out\_csv

    - Answer CSV: columns = demo, action list
    - GT CSV     : columns = demo, GT action list
    """
    df_gt  = pd.read_csv(gt_csv)
    df_ans = pd.read_csv(ans_csv)

    # column name check
    req_gt  = {"demo", "GT action list"}
    req_an  = {"demo", "action list"}
    miss_gt = req_gt - set(df_gt.columns)
    miss_an = req_an - set(df_ans.columns)
    if miss_gt:
        raise ValueError(f"Ground Truth CSV missing column: {miss_gt}")
    if miss_an:
        raise ValueError(f"Answer CSV missing column: {miss_an}")

    # Normalize demo identifiers
    df_gt["demo"]  = df_gt["demo"].astype(str).str.strip()
    df_ans["demo"] = df_ans["demo"].astype(str).str.strip()

    # If GT contains duplicate demos, keep only the last entry
    if df_gt["demo"].duplicated().any():
        df_gt = df_gt.sort_values("demo").drop_duplicates(subset=["demo"], keep="last")

    # Many-to-one merge: multiple answers are allowed; use the same GT for the same demo
    df = df_ans.merge(
        df_gt[["demo", "GT action list"]],
        on="demo",
        how="left",
        validate="many_to_one"
    )

    # Initialize result columns
    df["TSR"] = 0
    df["SSR"] = float("nan")
    df["FSR"] = 0
    df["Vision Error"] = 0
    df["Spatial Error"] = 0
    df["Temporal Error"] = 0

    for i, row in df.iterrows():
        pr_txt = row["action list"]
        gt_txt = row["GT action list"]
        if isinstance(pr_txt, str) and pr_txt.strip() and isinstance(gt_txt, str) and gt_txt.strip():
            try:
                res = evaluate_with_llm(gt_txt, pr_txt)
                df.at[i, "TSR"] = res["TSR"]
                df.at[i, "SSR"] = res["SSR"]
                df.at[i, "FSR"] = res["FSR"]
                df.at[i, "Vision Error"]   = res["Vision Error"]
                df.at[i, "Spatial Error"]  = res["Spatial Error"]
                df.at[i, "Temporal Error"] = res["Temporal Error"]
            except Exception:
                pass

    df.to_csv(out_csv, index=False)


# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser(description="Batch LLM-based evaluation for action lists (CSV).")
    ap.add_argument("mode", choices=["csv"], help="Only csv mode is supported in this script.")
    ap.add_argument("--gt_csv", required=True, help="Ground Truth CSV (columns: demo, GT action list)")
    ap.add_argument("--ans_csv", required=True, help="Answer CSV (columns: demo, action list)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    if args.mode == "csv":
        batch_eval_csv(args.gt_csv, args.ans_csv, args.out_csv)
        print(f"[OK] wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
