#!/usr/bin/env python3
"""
Compare model weights across V5 (transformers 4.35.1), V4 200ep (transformers 5.0),
production, and base pretrained to identify where V4's errors originate.

Key question: Since transformers 5.0's from_pretrained() calls _init_weights() 
AFTER loading (destroying certain weights), which layers diverge in V4?
"""

import torch
import numpy as np
from transformers import VitsModel, VitsConfig
from collections import defaultdict
import json

MODELS = {
    "base_pre": "mutisya/vits_luo_drL_24_5-v24_27_1_pre",
    "production": "mutisya/vits_luo_drL_24_5-v24_27_1_f",
    "v4_200ep": "mutisya/vits_luo_26_05_f_v4_200ep",
    "v5_t4351": "./outputs/vits_finetuned_luo_sharonm_v5_t4351",
}


def load_state_dicts():
    """Load all model state dicts."""
    state_dicts = {}
    for name, path in MODELS.items():
        print(f"Loading {name} from {path}...")
        model = VitsModel.from_pretrained(path)
        state_dicts[name] = {k: v.cpu().float() for k, v in model.state_dict().items()}
        del model
        torch.cuda.empty_cache()
    return state_dicts


def get_module_group(key):
    """Classify a weight key into its module group."""
    if key.startswith("text_encoder."):
        sub = key.split(".")[1] if len(key.split(".")) > 1 else ""
        if "emb" in sub:
            return "text_encoder.embedding"
        elif "encoder" in sub:
            return "text_encoder.encoder"
        elif "proj" in sub:
            return "text_encoder.proj"
        return "text_encoder.other"
    elif key.startswith("flow."):
        if "convs" in key or "conv" in key:
            return "flow.conv"
        elif "proj" in key or "linear" in key:
            return "flow.proj"
        return "flow.other"
    elif key.startswith("decoder."):
        if "resblock" in key:
            return "decoder.resblock"
        elif "conv_pre" in key:
            return "decoder.conv_pre"
        elif "conv_post" in key:
            return "decoder.conv_post"
        elif "upsampler" in key:
            return "decoder.upsampler"
        return "decoder.other"
    elif key.startswith("duration_predictor."):
        return "duration_predictor"
    elif key.startswith("posterior_encoder."):
        return "posterior_encoder"
    return "other"


def compare_weights(sd1, sd2, name1, name2):
    """Compare two state dicts, returning per-key metrics."""
    common_keys = sorted(set(sd1.keys()) & set(sd2.keys()))
    results = []
    
    for key in common_keys:
        w1 = sd1[key]
        w2 = sd2[key]
        
        if w1.shape != w2.shape:
            results.append({
                "key": key,
                "group": get_module_group(key),
                "shape": str(list(w1.shape)),
                "error": f"shape mismatch: {list(w1.shape)} vs {list(w2.shape)}",
            })
            continue
        
        diff = (w1 - w2).float()
        abs_diff = diff.abs()
        
        w_norm = w1.float().norm()
        rel_diff = diff.norm() / (w_norm + 1e-10)
        
        flat1 = w1.flatten().float()
        flat2 = w2.flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
        
        results.append({
            "key": key,
            "group": get_module_group(key),
            "shape": str(list(w1.shape)),
            "numel": w1.numel(),
            "l2_diff": float(diff.norm()),
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "rel_diff": float(rel_diff),
            "cosine_sim": float(cos_sim),
            f"{name1}_norm": float(w1.float().norm()),
            f"{name2}_norm": float(w2.float().norm()),
            f"{name1}_mean": float(w1.float().mean()),
            f"{name2}_mean": float(w2.float().mean()),
            f"{name1}_std": float(w1.float().std()),
            f"{name2}_std": float(w2.float().std()),
        })
    
    return results


def analyze_comparisons(results, name1, name2):
    """Analyze and print comparison results."""
    print(f"\n{'=' * 100}")
    print(f"  WEIGHT COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 100}")
    
    groups = defaultdict(list)
    for r in results:
        if "error" not in r:
            groups[r["group"]].append(r)
    
    print(f"\n{'Module Group':<30} {'Keys':>5} {'Mean RelDiff':>12} {'Max RelDiff':>12} {'Mean CosSim':>10} {'Params':>10}")
    print(f"{'-'*30} {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    
    group_summaries = {}
    for group in sorted(groups.keys()):
        items = groups[group]
        mean_rel = np.mean([r["rel_diff"] for r in items])
        max_rel = np.max([r["rel_diff"] for r in items])
        mean_cos = np.mean([r["cosine_sim"] for r in items])
        total_params = sum(r["numel"] for r in items)
        
        group_summaries[group] = {
            "n_keys": len(items),
            "mean_rel_diff": float(mean_rel),
            "max_rel_diff": float(max_rel),
            "mean_cosine_sim": float(mean_cos),
            "total_params": int(total_params),
        }
        
        flag = " <<<" if mean_rel > 0.1 or mean_cos < 0.99 else ""
        print(f"{group:<30} {len(items):>5} {mean_rel:>12.6f} {max_rel:>12.6f} {mean_cos:>10.6f} {total_params:>10,}{flag}")
    
    # Top 20 most different individual weights
    print(f"\n  Top 20 most different weights (by relative diff):")
    print(f"  {'Key':<65} {'RelDiff':>10} {'CosSim':>8} {'MaxAbsDiff':>10}")
    print(f"  {'-'*65} {'-'*10} {'-'*8} {'-'*10}")
    
    sorted_results = sorted([r for r in results if "error" not in r], key=lambda x: x["rel_diff"], reverse=True)
    for r in sorted_results[:20]:
        print(f"  {r['key']:<65} {r['rel_diff']:>10.6f} {r['cosine_sim']:>8.4f} {r['max_abs_diff']:>10.6f}")
    
    # Identity breakdown
    identical = [r for r in results if "error" not in r and r["cosine_sim"] > 0.9999]
    near_identical = [r for r in results if "error" not in r and 0.999 < r["cosine_sim"] <= 0.9999]
    different = [r for r in results if "error" not in r and r["cosine_sim"] <= 0.999]
    
    print(f"\n  Weight identity summary:")
    print(f"    Identical (cos > 0.9999):      {len(identical)}/{len(results)}")
    print(f"    Near-identical (0.999-0.9999):  {len(near_identical)}/{len(results)}")
    print(f"    Different (cos <= 0.999):       {len(different)}/{len(results)}")
    
    if different:
        print(f"\n  Weights with cosine_sim <= 0.999:")
        for r in sorted(different, key=lambda x: x["cosine_sim"]):
            print(f"    {r['key']:<65} cos={r['cosine_sim']:.6f} rel={r['rel_diff']:.6f}")
    
    return group_summaries


def check_init_weights_impact(sd_base, sd_v4, sd_v5):
    """
    Check if V4's deviations from base look like re-initialization.
    transformers 5.0 _init_weights() reinitializes:
    - nn.Linear: weight=N(0, config.initializer_range), bias=0
    - nn.Embedding: weight=N(0, config.initializer_range) 
    - nn.LayerNorm: weight=1, bias=0
    """
    print(f"\n{'=' * 100}")
    print(f"  INIT_WEIGHTS IMPACT ANALYSIS")
    print(f"  Checking if V4 weights look re-initialized vs base")
    print(f"{'=' * 100}")
    
    config = VitsConfig.from_pretrained(MODELS["base_pre"])
    init_range = config.initializer_range
    print(f"  initializer_range = {init_range}")
    
    suspect_keys = []
    
    for key in sorted(sd_base.keys()):
        if key not in sd_v4 or key not in sd_v5:
            continue
        
        w_base = sd_base[key].float()
        w_v4 = sd_v4[key].float()
        w_v5 = sd_v5[key].float()
        
        if w_base.shape != w_v4.shape or w_base.shape != w_v5.shape:
            continue
        
        diff_v4 = (w_v4 - w_base).norm() / (w_base.norm() + 1e-10)
        diff_v5 = (w_v5 - w_base).norm() / (w_base.norm() + 1e-10)
        
        v4_mean = w_v4.mean().item()
        v4_std = w_v4.std().item()
        
        # Flag if V4 diverged much more from base than V5 did
        if diff_v4 > 0.5 and diff_v4 > 3 * diff_v5:
            suspect_keys.append({
                "key": key,
                "group": get_module_group(key),
                "shape": str(list(w_base.shape)),
                "base_vs_v4_rel": float(diff_v4),
                "base_vs_v5_rel": float(diff_v5),
                "v4_mean": float(v4_mean),
                "v4_std": float(v4_std),
                "base_mean": float(w_base.mean()),
                "base_std": float(w_base.std()),
                "v5_mean": float(w_v5.mean()),
                "v5_std": float(w_v5.std()),
                "looks_reinitialized": abs(v4_mean) < 0.01 and abs(v4_std - init_range) < init_range * 0.5,
            })
    
    if suspect_keys:
        print(f"\n  Found {len(suspect_keys)} weights where V4 diverged much more from base than V5:")
        print(f"  {'Key':<60} {'Base→V4':>8} {'Base→V5':>8} {'V4 μ':>8} {'V4 σ':>8} {'Base μ':>8} {'Base σ':>8} {'ReInit?':>7}")
        print(f"  {'-'*60} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
        for s in sorted(suspect_keys, key=lambda x: x["base_vs_v4_rel"], reverse=True):
            reinit = "YES" if s["looks_reinitialized"] else "no"
            print(f"  {s['key']:<60} {s['base_vs_v4_rel']:>8.4f} {s['base_vs_v5_rel']:>8.4f} {s['v4_mean']:>8.4f} {s['v4_std']:>8.4f} {s['base_mean']:>8.4f} {s['base_std']:>8.4f} {reinit:>7}")
    else:
        print(f"\n  No weights found where V4 diverged significantly more than V5 from base.")
    
    # Check frozen weights (identical base<->model)
    for model_name, sd in [("V4", sd_v4), ("V5", sd_v5)]:
        frozen_count = 0
        frozen_groups = defaultdict(int)
        for key in sorted(sd_base.keys()):
            if key not in sd:
                continue
            if sd_base[key].shape != sd[key].shape:
                continue
            if torch.allclose(sd_base[key].float(), sd[key].float(), atol=1e-7):
                frozen_count += 1
                frozen_groups[get_module_group(key)] += 1
        
        print(f"\n  Weights identical between base and {model_name} (frozen/unchanged): {frozen_count}")
        for g, c in sorted(frozen_groups.items()):
            print(f"    {g}: {c}")
    
    return suspect_keys


def check_layer_norm_bias_patterns(sd_base, sd_v4, sd_v5):
    """
    Specifically check LayerNorm and bias parameters — _init_weights sets these to
    weight=1, bias=0. If V4 has these reset but V5 doesn't, that's the bug.
    """
    print(f"\n{'=' * 100}")
    print(f"  LAYERNORM & BIAS ANALYSIS")
    print(f"  _init_weights sets: LayerNorm.weight=1, LayerNorm.bias=0, Linear.bias=0")
    print(f"{'=' * 100}")
    
    # Find all layer_norm and bias keys
    for pattern_name, filter_fn in [
        ("LayerNorm weights", lambda k: "layer_norm" in k.lower() or "layernorm" in k.lower()),
        ("All bias parameters", lambda k: k.endswith(".bias")),
        ("Projection weights", lambda k: "proj" in k and k.endswith(".weight")),
    ]:
        print(f"\n  --- {pattern_name} ---")
        print(f"  {'Key':<60} {'Base→V4':>8} {'Base→V5':>8} {'V4≈Base':>7} {'V5≈Base':>7}")
        print(f"  {'-'*60} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        
        keys = [k for k in sorted(sd_base.keys()) if filter_fn(k) and k in sd_v4 and k in sd_v5]
        for key in keys:
            w_base = sd_base[key].float()
            w_v4 = sd_v4[key].float()
            w_v5 = sd_v5[key].float()
            
            if w_base.shape != w_v4.shape or w_base.shape != w_v5.shape:
                continue
            
            diff_v4 = (w_v4 - w_base).norm() / (w_base.norm() + 1e-10)
            diff_v5 = (w_v5 - w_base).norm() / (w_base.norm() + 1e-10)
            same_v4 = "YES" if torch.allclose(w_base, w_v4, atol=1e-5) else "no"
            same_v5 = "YES" if torch.allclose(w_base, w_v5, atol=1e-5) else "no"
            
            flag = " <<<" if same_v4 == "YES" and same_v5 != "YES" else ""
            flag = flag or (" ***" if diff_v4 > 2 * diff_v5 and diff_v4 > 0.1 else "")
            print(f"  {key:<60} {diff_v4:>8.4f} {diff_v5:>8.4f} {same_v4:>7} {same_v5:>7}{flag}")


def main():
    print("Loading all model state dicts...")
    sds = load_state_dicts()
    
    # 1. V5 vs V4 — main comparison
    results_v5_v4 = compare_weights(sds["v5_t4351"], sds["v4_200ep"], "v5", "v4")
    gs_v5_v4 = analyze_comparisons(results_v5_v4, "v5_t4351", "v4_200ep")
    
    # 2. V5 vs Production  
    results_v5_prod = compare_weights(sds["v5_t4351"], sds["production"], "v5", "prod")
    gs_v5_prod = analyze_comparisons(results_v5_prod, "v5_t4351", "production")
    
    # 3. V4 vs Base
    results_v4_base = compare_weights(sds["v4_200ep"], sds["base_pre"], "v4", "base")
    gs_v4_base = analyze_comparisons(results_v4_base, "v4_200ep", "base_pre")
    
    # 4. V5 vs Base
    results_v5_base = compare_weights(sds["v5_t4351"], sds["base_pre"], "v5", "base")
    gs_v5_base = analyze_comparisons(results_v5_base, "v5_t4351", "base_pre")
    
    # 5. Init weights impact analysis
    suspect = check_init_weights_impact(sds["base_pre"], sds["v4_200ep"], sds["v5_t4351"])
    
    # 6. LayerNorm & bias specific check
    check_layer_norm_bias_patterns(sds["base_pre"], sds["v4_200ep"], sds["v5_t4351"])
    
    # 7. Cross-model group comparison table
    print(f"\n\n{'=' * 100}")
    print(f"  CROSS-MODEL GROUP COMPARISON (relative diff from base)")
    print(f"{'=' * 100}")
    
    all_groups = sorted(set(list(gs_v4_base.keys()) + list(gs_v5_base.keys())))
    print(f"  {'Module Group':<30} {'V4→Base RelDiff':>15} {'V5→Base RelDiff':>15} {'V4→Base CosSim':>15} {'V5→Base CosSim':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    for g in all_groups:
        v4b = gs_v4_base.get(g, {})
        v5b = gs_v5_base.get(g, {})
        v4_rel = v4b.get("mean_rel_diff", 0)
        v5_rel = v5b.get("mean_rel_diff", 0)
        v4_cos = v4b.get("mean_cosine_sim", 1)
        v5_cos = v5b.get("mean_cosine_sim", 1)
        flag = " <<<" if abs(v4_rel - v5_rel) > 0.05 else ""
        print(f"  {g:<30} {v4_rel:>15.6f} {v5_rel:>15.6f} {v4_cos:>15.6f} {v5_cos:>15.6f}{flag}")
    
    # Save all results
    all_results = {
        "v5_vs_v4": results_v5_v4,
        "v5_vs_prod": results_v5_prod,
        "v4_vs_base": results_v4_base,
        "v5_vs_base": results_v5_base,
        "suspect_reinit": suspect,
    }
    with open("weight_comparison_v5_v4.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to weight_comparison_v5_v4.json")


if __name__ == "__main__":
    main()
