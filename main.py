import json
import math
import os
import sys
from pathlib import Path

# Path setup & environment normalization
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

def clamp_p(p):
    return max(0.0001, min(0.9999, p))

def logit(p):
    cp = clamp_p(p)
    return math.log(cp / (1.0 - cp))

def expit(x):
    if x > 30.0:
        return 1.0 - 1e-12
    if x < -30.0:
        return 1e-12
    return 1.0 / (1.0 + math.exp(-x))

def solve_case(case_data):
    trophies = case_data.get("trophies", [])
    history = case_data.get("history", [])
    horizon_minutes = float(case_data.get("horizon_minutes", 60))
    d_days = max(1e-4, horizon_minutes / 1440.0)
    
    # Extract last known valid rates
    last_rates = {}
    history_by_trophy = {t["id"]: [] for t in trophies}
    
    for read in history:
        rates = read.get("rates", [])
        for i, t in enumerate(trophies):
            t_id = t["id"]
            if i < len(rates) and rates[i] is not None:
                val = float(rates[i])
                last_rates[t_id] = val
                history_by_trophy[t_id].append(val)
                
    results = []
    
    for t in trophies:
        t_id = t["id"]
        tier = t.get("tier", "bronze").lower()
        secret = bool(t.get("secret", False))
        
        # Base rate
        r_last = last_rates.get(t_id, 1.0)
        p_last = r_last / 100.0
        base_logit = logit(p_last)
        
        # Historical drift fit
        h_vals = history_by_trophy.get(t_id, [])
        trend_drift = 0.0
        if len(h_vals) >= 3:
            v_start = h_vals[0] / 100.0
            v_end = h_vals[-1] / 100.0
            raw_slope = (logit(v_end) - logit(v_start)) / max(1, len(h_vals) - 1)
            # Damped trend projection
            trend_drift = raw_slope * min(5.0, math.sqrt(d_days))
            
        # Tier prior adjustment
        tier_adj = 0.0
        if tier == "platinum":
            tier_adj = -0.15 * math.sqrt(d_days)
        elif tier == "gold":
            tier_adj = -0.05 * math.sqrt(d_days)
        elif tier == "bronze" and p_last > 0.4:
            tier_adj = 0.05 * math.sqrt(d_days)
            
        if secret:
            tier_adj -= 0.05 * math.sqrt(d_days)
            
        # Predicted median logit
        pred_median_logit = base_logit + trend_drift + tier_adj
        
        # Volatility / sigma scale
        sigma = 0.45 * math.sqrt(d_days)
        if tier == "platinum":
            sigma = 0.35 * math.sqrt(d_days)
            
        quantiles = []
        for z_val in Z:
            q_logit = pred_median_logit + z_val * sigma
            q_pct = round(expit(q_logit) * 100.0, 4)
            quantiles.append(q_pct)
            
        # Ensure monotonic non-decreasing
        for j in range(1, len(quantiles)):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
                
        results.append({
            "id": t_id,
            "q": quantiles
        })
        
    return {"trophies": results}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    in_files = sorted(list(INPUT_DIR.glob("*.in")))
    if not in_files:
        print(f"No *.in files found in {INPUT_DIR}")
        return
        
    for in_file in in_files:
        with open(in_file, "r", encoding="utf-8") as f:
            case_data = json.load(f)
            
        out_data = solve_case(case_data)
        out_file = OUTPUT_DIR / f"{in_file.stem}.out"
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, separators=(",", ":"))
            
        print(f"Wrote {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
