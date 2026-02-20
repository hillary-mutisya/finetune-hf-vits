"""
Deep comparison of VITS model weights and structure between:
  - mutisya/vits_luo_26_05_f_v1 (retrained, 200 epochs - noisy)
  - mutisya/vits_luo_drL_24_5-v24_27_1_f (production - cleaner)
"""

import torch
import json
import numpy as np
from transformers import VitsModel, VitsTokenizer, AutoConfig
from collections import OrderedDict

RETRAINED = "mutisya/vits_luo_26_05_f_v1"
PRODUCTION = "mutisya/vits_luo_drL_24_5-v24_27_1_f"

print("=" * 80)
print("LOADING MODELS")
print("=" * 80)

print(f"\nLoading retrained: {RETRAINED}")
config_r = AutoConfig.from_pretrained(RETRAINED)
model_r = VitsModel.from_pretrained(RETRAINED)
tok_r = VitsTokenizer.from_pretrained(RETRAINED)

print(f"\nLoading production: {PRODUCTION}")
config_p = AutoConfig.from_pretrained(PRODUCTION)
model_p = VitsModel.from_pretrained(PRODUCTION)
tok_p = VitsTokenizer.from_pretrained(PRODUCTION)

# ── 1. CONFIG COMPARISON ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1. CONFIG COMPARISON")
print("=" * 80)

cfg_r = config_r.to_dict()
cfg_p = config_p.to_dict()

all_keys = sorted(set(list(cfg_r.keys()) + list(cfg_p.keys())))
diffs = []
for k in all_keys:
    v_r = cfg_r.get(k, "<MISSING>")
    v_p = cfg_p.get(k, "<MISSING>")
    if v_r != v_p:
        diffs.append((k, v_r, v_p))

if diffs:
    print(f"\n{'Key':<45} {'Retrained':<30} {'Production':<30}")
    print("-" * 105)
    for k, vr, vp in diffs:
        vr_s = str(vr)[:28]
        vp_s = str(vp)[:28]
        print(f"{k:<45} {vr_s:<30} {vp_s:<30}")
else:
    print("\nConfigs are IDENTICAL")

# ── 2. TOKENIZER / VOCAB COMPARISON ──────────────────────────────────────────
print("\n" + "=" * 80)
print("2. TOKENIZER / VOCAB COMPARISON")
print("=" * 80)

vocab_r = tok_r.get_vocab()
vocab_p = tok_p.get_vocab()
print(f"Retrained vocab size: {len(vocab_r)}")
print(f"Production vocab size: {len(vocab_p)}")

only_r = set(vocab_r.keys()) - set(vocab_p.keys())
only_p = set(vocab_p.keys()) - set(vocab_r.keys())
if only_r:
    print(f"Only in retrained ({len(only_r)}): {sorted(only_r)[:20]}")
if only_p:
    print(f"Only in production ({len(only_p)}): {sorted(only_p)[:20]}")

# Check if same chars map to same IDs
common = set(vocab_r.keys()) & set(vocab_p.keys())
id_mismatches = []
for c in sorted(common):
    if vocab_r[c] != vocab_p[c]:
        id_mismatches.append((c, vocab_r[c], vocab_p[c]))
if id_mismatches:
    print(f"\nID mismatches for common tokens ({len(id_mismatches)}):")
    for c, ir, ip in id_mismatches[:20]:
        print(f"  '{c}': retrained={ir}, production={ip}")
else:
    print("All common tokens have matching IDs")

# ── 3. PARAMETER STRUCTURE COMPARISON ─────────────────────────────────────────
print("\n" + "=" * 80)
print("3. PARAMETER STRUCTURE COMPARISON")
print("=" * 80)

sd_r = model_r.state_dict()
sd_p = model_p.state_dict()

keys_r = set(sd_r.keys())
keys_p = set(sd_p.keys())

only_in_r = keys_r - keys_p
only_in_p = keys_p - keys_r
common_keys = sorted(keys_r & keys_p)

print(f"\nRetrained params: {len(keys_r)}")
print(f"Production params: {len(keys_p)}")
print(f"Common params: {len(common_keys)}")

if only_in_r:
    print(f"\nOnly in retrained ({len(only_in_r)}):")
    for k in sorted(only_in_r):
        print(f"  {k}: {sd_r[k].shape}")
if only_in_p:
    print(f"\nOnly in production ({len(only_in_p)}):")
    for k in sorted(only_in_p):
        print(f"  {k}: {sd_p[k].shape}")

# Check shape mismatches
shape_mismatches = []
for k in common_keys:
    if sd_r[k].shape != sd_p[k].shape:
        shape_mismatches.append((k, sd_r[k].shape, sd_p[k].shape))

if shape_mismatches:
    print(f"\nShape mismatches ({len(shape_mismatches)}):")
    for k, sr, sp in shape_mismatches:
        print(f"  {k}: retrained={sr}, production={sp}")
else:
    print("\nAll common parameters have matching shapes")

# ── 4. WEIGHT STATISTICS COMPARISON ──────────────────────────────────────────
print("\n" + "=" * 80)
print("4. WEIGHT STATISTICS COMPARISON (per-layer)")
print("=" * 80)

# Group parameters by module
def get_module(key):
    parts = key.split(".")
    # Group by top 2-3 levels
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return ".".join(parts[:2])

# Compute per-parameter stats
param_stats = []
for k in common_keys:
    if sd_r[k].shape != sd_p[k].shape:
        continue
    
    wr = sd_r[k].float().cpu()
    wp = sd_p[k].float().cpu()
    
    diff = wr - wp
    
    stats = {
        "key": k,
        "module": get_module(k),
        "shape": tuple(wr.shape),
        "numel": wr.numel(),
        # Retrained stats
        "r_mean": wr.mean().item(),
        "r_std": wr.std().item(),
        "r_min": wr.min().item(),
        "r_max": wr.max().item(),
        "r_norm": wr.norm().item(),
        "r_abs_mean": wr.abs().mean().item(),
        # Production stats
        "p_mean": wp.mean().item(),
        "p_std": wp.std().item(),
        "p_min": wp.min().item(),
        "p_max": wp.max().item(),
        "p_norm": wp.norm().item(),
        "p_abs_mean": wp.abs().mean().item(),
        # Difference stats
        "diff_mean": diff.mean().item(),
        "diff_std": diff.std().item(),
        "diff_abs_mean": diff.abs().mean().item(),
        "diff_max": diff.abs().max().item(),
        "diff_norm": diff.norm().item(),
        # Cosine similarity
        "cosine_sim": torch.nn.functional.cosine_similarity(
            wr.reshape(1, -1), wp.reshape(1, -1)
        ).item() if wr.numel() > 1 else float('nan'),
        # Relative difference
        "rel_diff": (diff.norm() / (wp.norm() + 1e-8)).item(),
    }
    param_stats.append(stats)

# Print most divergent parameters
print("\nTop 30 most divergent parameters (by relative difference):")
print(f"{'Parameter':<65} {'RelDiff':>8} {'CosSim':>8} {'R_std':>8} {'P_std':>8}")
print("-" * 100)
sorted_by_diff = sorted(param_stats, key=lambda x: x["rel_diff"], reverse=True)
for s in sorted_by_diff[:30]:
    print(f"{s['key']:<65} {s['rel_diff']:>8.4f} {s['cosine_sim']:>8.4f} {s['r_std']:>8.4f} {s['p_std']:>8.4f}")

# ── 5. MODULE-LEVEL AGGREGATED COMPARISON ─────────────────────────────────────
print("\n" + "=" * 80)
print("5. MODULE-LEVEL AGGREGATED COMPARISON")
print("=" * 80)

# Aggregate by top-level module
def get_top_module(key):
    parts = key.split(".")
    return parts[0]  # e.g., "decoder", "flow", "text_encoder", etc.

module_stats = {}
for s in param_stats:
    mod = get_top_module(s["key"])
    if mod not in module_stats:
        module_stats[mod] = {
            "count": 0, "total_params": 0,
            "total_rel_diff": 0, "total_cosine": 0,
            "max_rel_diff": 0, "min_cosine": 1.0,
            "r_norms": [], "p_norms": [],
        }
    ms = module_stats[mod]
    ms["count"] += 1
    ms["total_params"] += s["numel"]
    ms["total_rel_diff"] += s["rel_diff"]
    if not np.isnan(s["cosine_sim"]):
        ms["total_cosine"] += s["cosine_sim"]
    ms["max_rel_diff"] = max(ms["max_rel_diff"], s["rel_diff"])
    if not np.isnan(s["cosine_sim"]):
        ms["min_cosine"] = min(ms["min_cosine"], s["cosine_sim"])
    ms["r_norms"].append(s["r_norm"])
    ms["p_norms"].append(s["p_norm"])

print(f"\n{'Module':<25} {'#Params':>10} {'#Layers':>8} {'AvgRelDiff':>11} {'MaxRelDiff':>11} {'MinCosSim':>10} {'AvgR_norm':>10} {'AvgP_norm':>10}")
print("-" * 95)
for mod in sorted(module_stats.keys()):
    ms = module_stats[mod]
    avg_rd = ms["total_rel_diff"] / ms["count"]
    avg_cos = ms["total_cosine"] / ms["count"]
    avg_rn = np.mean(ms["r_norms"])
    avg_pn = np.mean(ms["p_norms"])
    print(f"{mod:<25} {ms['total_params']:>10,} {ms['count']:>8} {avg_rd:>11.4f} {ms['max_rel_diff']:>11.4f} {ms['min_cosine']:>10.4f} {avg_rn:>10.2f} {avg_pn:>10.2f}")

# ── 6. DECODER (VOCODER) DEEP DIVE ──────────────────────────────────────────
print("\n" + "=" * 80)
print("6. DECODER (VOCODER / HiFi-GAN) DEEP DIVE")
print("=" * 80)

decoder_stats = [s for s in param_stats if s["key"].startswith("decoder.")]
print(f"\nDecoder has {len(decoder_stats)} parameter tensors")

# Sub-module breakdown
decoder_submods = {}
for s in decoder_stats:
    parts = s["key"].split(".")
    if len(parts) >= 3:
        submod = ".".join(parts[1:3])
    else:
        submod = parts[1]
    if submod not in decoder_submods:
        decoder_submods[submod] = {"params": 0, "rel_diffs": [], "cosines": []}
    decoder_submods[submod]["params"] += s["numel"]
    decoder_submods[submod]["rel_diffs"].append(s["rel_diff"])
    if not np.isnan(s["cosine_sim"]):
        decoder_submods[submod]["cosines"].append(s["cosine_sim"])

print(f"\n{'Decoder Sub-module':<40} {'Params':>10} {'AvgRelDiff':>11} {'MaxRelDiff':>11} {'MinCosSim':>10}")
print("-" * 85)
for submod in sorted(decoder_submods.keys()):
    ds = decoder_submods[submod]
    avg_rd = np.mean(ds["rel_diffs"])
    max_rd = np.max(ds["rel_diffs"])
    min_cos = np.min(ds["cosines"]) if ds["cosines"] else float('nan')
    print(f"{submod:<40} {ds['params']:>10,} {avg_rd:>11.4f} {max_rd:>11.4f} {min_cos:>10.4f}")

# ── 7. FLOW MODULE DEEP DIVE ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("7. FLOW MODULE DEEP DIVE")
print("=" * 80)

flow_stats = [s for s in param_stats if s["key"].startswith("flow.")]
print(f"\nFlow has {len(flow_stats)} parameter tensors")

flow_submods = {}
for s in flow_stats:
    parts = s["key"].split(".")
    if len(parts) >= 4:
        submod = ".".join(parts[1:4])
    else:
        submod = ".".join(parts[1:3])
    if submod not in flow_submods:
        flow_submods[submod] = {"params": 0, "rel_diffs": [], "cosines": []}
    flow_submods[submod]["params"] += s["numel"]
    flow_submods[submod]["rel_diffs"].append(s["rel_diff"])
    if not np.isnan(s["cosine_sim"]):
        flow_submods[submod]["cosines"].append(s["cosine_sim"])

print(f"\n{'Flow Sub-module':<40} {'Params':>10} {'AvgRelDiff':>11} {'MaxRelDiff':>11} {'MinCosSim':>10}")
print("-" * 85)
for submod in sorted(flow_submods.keys()):
    ds = flow_submods[submod]
    avg_rd = np.mean(ds["rel_diffs"])
    max_rd = np.max(ds["rel_diffs"])
    min_cos = np.min(ds["cosines"]) if ds["cosines"] else float('nan')
    print(f"{submod:<40} {ds['params']:>10,} {avg_rd:>11.4f} {max_rd:>11.4f} {min_cos:>10.4f}")

# ── 8. WEIGHT MAGNITUDE ANALYSIS ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("8. WEIGHT MAGNITUDE ANALYSIS - Are retrained weights 'blown up'?")
print("=" * 80)

# Check if retrained model has larger weight magnitudes (sign of overtraining)
for mod in sorted(module_stats.keys()):
    r_total_norm = 0.0
    p_total_norm = 0.0
    for s in param_stats:
        if get_top_module(s["key"]) == mod:
            r_total_norm += s["r_norm"] ** 2
            p_total_norm += s["p_norm"] ** 2
    r_total_norm = r_total_norm ** 0.5
    p_total_norm = p_total_norm ** 0.5
    ratio = r_total_norm / (p_total_norm + 1e-8)
    indicator = "⚠️" if abs(ratio - 1.0) > 0.1 else "✓"
    print(f"  {mod:<25} R_norm={r_total_norm:>10.2f}  P_norm={p_total_norm:>10.2f}  Ratio={ratio:.4f}  {indicator}")

# ── 9. SPECIFIC WEIGHT DISTRIBUTION ANALYSIS ─────────────────────────────────
print("\n" + "=" * 80)
print("9. WEIGHT DISTRIBUTION ANALYSIS (key layers)")
print("=" * 80)

# Analyze specific important layers
key_layers = [
    "decoder.input_conv.weight",
    "decoder.input_conv.bias",
    "decoder.apply_weight_norm_orig_freq.upsampler.0.weight",
    "decoder.apply_weight_norm_orig_freq.upsampler.0.bias",
]

# Find actual decoder upsampler layer names
for k in common_keys:
    if "upsampl" in k.lower() and "weight" in k and k.startswith("decoder"):
        if k not in key_layers:
            key_layers.append(k)
    if "resblock" in k.lower() and "weight" in k and k.startswith("decoder"):
        # Just add first few
        if len([x for x in key_layers if "resblock" in x]) < 6:
            key_layers.append(k)

# Also check output conv
for k in common_keys:
    if "output" in k.lower() and k.startswith("decoder"):
        key_layers.append(k)

for k in key_layers:
    if k not in sd_r or k not in sd_p:
        continue
    if sd_r[k].shape != sd_p[k].shape:
        continue
    
    wr = sd_r[k].float().cpu()
    wp = sd_p[k].float().cpu()
    diff = wr - wp
    
    print(f"\n  {k} {list(wr.shape)}")
    print(f"    Retrained:  mean={wr.mean():.6f} std={wr.std():.6f} min={wr.min():.6f} max={wr.max():.6f}")
    print(f"    Production: mean={wp.mean():.6f} std={wp.std():.6f} min={wp.min():.6f} max={wp.max():.6f}")
    print(f"    Difference: mean={diff.mean():.6f} std={diff.std():.6f} |max|={diff.abs().max():.6f}")
    
    # Check for outlier weights
    r_outliers = (wr.abs() > 3 * wr.std()).sum().item()
    p_outliers = (wp.abs() > 3 * wp.std()).sum().item()
    print(f"    Outliers (>3σ): retrained={r_outliers}/{wr.numel()} production={p_outliers}/{wp.numel()}")

# ── 10. POSTERIOR ENCODER & DURATION PREDICTOR ────────────────────────────────
print("\n" + "=" * 80)
print("10. POSTERIOR ENCODER & DURATION PREDICTOR COMPARISON")
print("=" * 80)

for prefix in ["posterior_encoder", "duration_predictor"]:
    sub_stats = [s for s in param_stats if s["key"].startswith(prefix)]
    if not sub_stats:
        continue
    
    total_r_norm = sum(s["r_norm"]**2 for s in sub_stats) ** 0.5
    total_p_norm = sum(s["p_norm"]**2 for s in sub_stats) ** 0.5
    avg_cosine = np.mean([s["cosine_sim"] for s in sub_stats if not np.isnan(s["cosine_sim"])])
    avg_rel = np.mean([s["rel_diff"] for s in sub_stats])
    max_rel = max(s["rel_diff"] for s in sub_stats)
    
    print(f"\n  {prefix}:")
    print(f"    Layers: {len(sub_stats)}")
    print(f"    Total norm: R={total_r_norm:.4f}  P={total_p_norm:.4f}  ratio={total_r_norm/(total_p_norm+1e-8):.4f}")
    print(f"    Avg cosine sim: {avg_cosine:.4f}")
    print(f"    Avg relative diff: {avg_rel:.4f}  max: {max_rel:.4f}")

# ── 11. CHECK FOR NaN/Inf ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("11. NaN/Inf CHECK")
print("=" * 80)

for label, sd in [("retrained", sd_r), ("production", sd_p)]:
    nan_keys = []
    inf_keys = []
    for k, v in sd.items():
        if torch.isnan(v.float()).any():
            nan_keys.append(k)
        if torch.isinf(v.float()).any():
            inf_keys.append(k)
    if nan_keys:
        print(f"  {label}: NaN found in {nan_keys}")
    else:
        print(f"  {label}: No NaN values ✓")
    if inf_keys:
        print(f"  {label}: Inf found in {inf_keys}")
    else:
        print(f"  {label}: No Inf values ✓")

# ── 12. EMBEDDING COMPARISON ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("12. TEXT ENCODER EMBEDDING COMPARISON")
print("=" * 80)

for k in common_keys:
    if "embed" in k.lower() and k.startswith("text_encoder"):
        if sd_r[k].shape != sd_p[k].shape:
            print(f"\n  {k}: SHAPE MISMATCH r={sd_r[k].shape} p={sd_p[k].shape}")
            continue
        wr = sd_r[k].float()
        wp = sd_p[k].float()
        diff = wr - wp
        
        # Check per-token embedding differences
        if wr.dim() == 2:
            per_token_diff = diff.norm(dim=1)
            top_diff_tokens = per_token_diff.topk(min(10, len(per_token_diff)))
            print(f"\n  {k} {list(wr.shape)}")
            print(f"    Overall cosine sim: {torch.nn.functional.cosine_similarity(wr.reshape(1,-1), wp.reshape(1,-1)).item():.4f}")
            print(f"    Top divergent token indices: {top_diff_tokens.indices.tolist()}")
            print(f"    Their diff norms: {[f'{v:.4f}' for v in top_diff_tokens.values.tolist()]}")

# ── 13. SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("13. OVERALL SUMMARY")
print("=" * 80)

total_r = sum(s["r_norm"]**2 for s in param_stats) ** 0.5
total_p = sum(s["p_norm"]**2 for s in param_stats) ** 0.5
avg_cosine_all = np.mean([s["cosine_sim"] for s in param_stats if not np.isnan(s["cosine_sim"])])
avg_rel_all = np.mean([s["rel_diff"] for s in param_stats])

print(f"\n  Total weight norm: retrained={total_r:.2f}  production={total_p:.2f}  ratio={total_r/total_p:.4f}")
print(f"  Average cosine similarity across all layers: {avg_cosine_all:.4f}")
print(f"  Average relative difference: {avg_rel_all:.4f}")

# Count layers that are nearly identical vs very different
identical = sum(1 for s in param_stats if s["rel_diff"] < 0.01)
similar = sum(1 for s in param_stats if 0.01 <= s["rel_diff"] < 0.1)
different = sum(1 for s in param_stats if 0.1 <= s["rel_diff"] < 0.5)
very_diff = sum(1 for s in param_stats if s["rel_diff"] >= 0.5)

print(f"\n  Layer divergence distribution:")
print(f"    Nearly identical (rel_diff < 0.01): {identical}")
print(f"    Similar (0.01-0.1): {similar}")
print(f"    Different (0.1-0.5): {different}")
print(f"    Very different (>0.5): {very_diff}")

# Which modules are most different?
print(f"\n  Most problematic modules (highest avg relative diff):")
mod_avg = {}
for mod in module_stats:
    ms = module_stats[mod]
    mod_avg[mod] = ms["total_rel_diff"] / ms["count"]
for mod, avg in sorted(mod_avg.items(), key=lambda x: x[1], reverse=True):
    print(f"    {mod:<25} avg_rel_diff={avg:.4f}")

print("\nDone!")
