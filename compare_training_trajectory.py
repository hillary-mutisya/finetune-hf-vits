"""
Compare the base/pretrained model with both the production and retrained models
to understand the training trajectory differences.

Base:       mutisya/vits_luo_drL_24_5-v24_27_1_pre
Production: mutisya/vits_luo_drL_24_5-v24_27_1_f
Retrained:  mutisya/vits_luo_26_05_f_v1
"""

import torch
import numpy as np
from transformers import VitsModel, AutoConfig

BASE = "mutisya/vits_luo_drL_24_5-v24_27_1_pre"
PRODUCTION = "mutisya/vits_luo_drL_24_5-v24_27_1_f"
RETRAINED = "mutisya/vits_luo_26_05_f_v1"

print("Loading base model...")
model_b = VitsModel.from_pretrained(BASE)
print("Loading production model...")
model_p = VitsModel.from_pretrained(PRODUCTION)
print("Loading retrained model...")
model_r = VitsModel.from_pretrained(RETRAINED)

sd_b = model_b.state_dict()
sd_p = model_p.state_dict()
sd_r = model_r.state_dict()

# Check if base has same keys
common = sorted(set(sd_b.keys()) & set(sd_p.keys()) & set(sd_r.keys()))
print(f"\nBase params: {len(sd_b)}, Production: {len(sd_p)}, Retrained: {len(sd_r)}")
print(f"Common params: {len(common)}")

# Check for shape differences with base
shape_diff_bp = [(k, sd_b[k].shape, sd_p[k].shape) for k in common if sd_b[k].shape != sd_p[k].shape]
shape_diff_br = [(k, sd_b[k].shape, sd_r[k].shape) for k in common if sd_b[k].shape != sd_r[k].shape]
if shape_diff_bp:
    print(f"\nBase vs Production shape diffs: {len(shape_diff_bp)}")
    for k, sb, sp in shape_diff_bp[:5]:
        print(f"  {k}: base={sb} prod={sp}")
if shape_diff_br:
    print(f"\nBase vs Retrained shape diffs: {len(shape_diff_br)}")
    for k, sb, sr in shape_diff_br[:5]:
        print(f"  {k}: base={sb} retr={sr}")

# ── Key question: Did production start from the same base as retrained? ──
print("\n" + "=" * 80)
print("TRAINING TRAJECTORY ANALYSIS")
print("Which model diverged more from the base?")
print("=" * 80)

def get_top_module(key):
    return key.split(".")[0]

modules = sorted(set(get_top_module(k) for k in common))

print(f"\n{'Module':<25} {'Base→Prod':>10} {'Base→Retr':>10} {'Prod→Retr':>10} {'B→P cos':>8} {'B→R cos':>8} {'P→R cos':>8}")
print("-" * 100)

all_bp_diffs = []
all_br_diffs = []
all_pr_diffs = []

for mod in modules:
    mod_keys = [k for k in common if get_top_module(k) == mod and sd_b[k].shape == sd_p[k].shape == sd_r[k].shape]
    
    bp_norms = []
    br_norms = []
    pr_norms = []
    bp_cosines = []
    br_cosines = []
    pr_cosines = []
    
    for k in mod_keys:
        wb = sd_b[k].float().cpu()
        wp = sd_p[k].float().cpu()
        wr = sd_r[k].float().cpu()
        
        bp = wp - wb
        br = wr - wb
        pr = wr - wp
        
        bp_rel = bp.norm().item() / (wb.norm().item() + 1e-8)
        br_rel = br.norm().item() / (wb.norm().item() + 1e-8)
        pr_rel = pr.norm().item() / (wp.norm().item() + 1e-8)
        
        bp_norms.append(bp_rel)
        br_norms.append(br_rel)
        pr_norms.append(pr_rel)
        
        if wb.numel() > 1:
            bp_cosines.append(torch.nn.functional.cosine_similarity(wb.reshape(1,-1), wp.reshape(1,-1)).item())
            br_cosines.append(torch.nn.functional.cosine_similarity(wb.reshape(1,-1), wr.reshape(1,-1)).item())
            pr_cosines.append(torch.nn.functional.cosine_similarity(wp.reshape(1,-1), wr.reshape(1,-1)).item())
        
        all_bp_diffs.append(bp_rel)
        all_br_diffs.append(br_rel)
        all_pr_diffs.append(pr_rel)
    
    avg_bp = np.mean(bp_norms)
    avg_br = np.mean(br_norms)
    avg_pr = np.mean(pr_norms)
    avg_bp_cos = np.mean(bp_cosines) if bp_cosines else float('nan')
    avg_br_cos = np.mean(br_cosines) if br_cosines else float('nan')
    avg_pr_cos = np.mean(pr_cosines) if pr_cosines else float('nan')
    
    print(f"{mod:<25} {avg_bp:>10.4f} {avg_br:>10.4f} {avg_pr:>10.4f} {avg_bp_cos:>8.4f} {avg_br_cos:>8.4f} {avg_pr_cos:>8.4f}")

print(f"\n{'OVERALL':<25} {np.mean(all_bp_diffs):>10.4f} {np.mean(all_br_diffs):>10.4f} {np.mean(all_pr_diffs):>10.4f}")

# ── Check if any layers are FROZEN (identical to base) ──
print("\n" + "=" * 80)
print("FROZEN LAYER CHECK")
print("Which layers didn't change from base?")
print("=" * 80)

frozen_in_prod = []
frozen_in_retr = []
for k in common:
    if sd_b[k].shape != sd_p[k].shape or sd_b[k].shape != sd_r[k].shape:
        continue
    wb = sd_b[k].float()
    wp = sd_p[k].float()
    wr = sd_r[k].float()
    
    bp_identical = torch.equal(wb, wp) or (wb - wp).abs().max().item() < 1e-7
    br_identical = torch.equal(wb, wr) or (wb - wr).abs().max().item() < 1e-7
    
    if bp_identical:
        frozen_in_prod.append(k)
    if br_identical:
        frozen_in_retr.append(k)

print(f"\nLayers frozen (unchanged from base) in production: {len(frozen_in_prod)}/{len(common)}")
if frozen_in_prod:
    # Show which modules
    frozen_mods_p = {}
    for k in frozen_in_prod:
        mod = get_top_module(k)
        frozen_mods_p[mod] = frozen_mods_p.get(mod, 0) + 1
    for mod, count in sorted(frozen_mods_p.items()):
        total = sum(1 for k in common if get_top_module(k) == mod)
        print(f"  {mod}: {count}/{total} layers frozen")

print(f"\nLayers frozen (unchanged from base) in retrained: {len(frozen_in_retr)}/{len(common)}")
if frozen_in_retr:
    frozen_mods_r = {}
    for k in frozen_in_retr:
        mod = get_top_module(k)
        frozen_mods_r[mod] = frozen_mods_r.get(mod, 0) + 1
    for mod, count in sorted(frozen_mods_r.items()):
        total = sum(1 for k in common if get_top_module(k) == mod)
        print(f"  {mod}: {count}/{total} layers frozen")

# ── Check text encoder specifically ──
print("\n" + "=" * 80)
print("TEXT ENCODER DEEP COMPARISON (most divergent module)")
print("=" * 80)

embed_b = sd_b["text_encoder.embed_tokens.weight"].float()
embed_p = sd_p["text_encoder.embed_tokens.weight"].float()
embed_r = sd_r["text_encoder.embed_tokens.weight"].float()

print(f"\nEmbedding shapes: base={embed_b.shape}, prod={embed_p.shape}, retr={embed_r.shape}")

if embed_b.shape == embed_p.shape == embed_r.shape:
    # Per-token analysis
    print(f"\nPer-token embedding distance from base:")
    print(f"{'Token':>5} {'Base→Prod':>12} {'Base→Retr':>12} {'Prod→Retr':>12} {'B→P cos':>10} {'B→R cos':>10}")
    print("-" * 65)
    for i in range(embed_b.shape[0]):
        eb = embed_b[i]
        ep = embed_p[i]
        er = embed_r[i]
        
        bp_dist = (ep - eb).norm().item()
        br_dist = (er - eb).norm().item()
        pr_dist = (er - ep).norm().item()
        bp_cos = torch.nn.functional.cosine_similarity(eb.unsqueeze(0), ep.unsqueeze(0)).item()
        br_cos = torch.nn.functional.cosine_similarity(eb.unsqueeze(0), er.unsqueeze(0)).item()
        
        print(f"{i:>5} {bp_dist:>12.4f} {br_dist:>12.4f} {pr_dist:>12.4f} {bp_cos:>10.4f} {br_cos:>10.4f}")

# ── Decoder upsampler comparison with base ──
print("\n" + "=" * 80)
print("DECODER UPSAMPLER COMPARISON WITH BASE")
print("=" * 80)

for i in range(4):
    wkey = f"decoder.upsampler.{i}.weight"
    bkey = f"decoder.upsampler.{i}.bias"
    
    for k in [wkey, bkey]:
        if k not in sd_b or k not in sd_p or k not in sd_r:
            continue
        wb = sd_b[k].float()
        wp = sd_p[k].float()
        wr = sd_r[k].float()
        
        if wb.shape != wp.shape or wb.shape != wr.shape:
            print(f"  {k}: Shape mismatch!")
            continue
        
        bp_rel = (wp - wb).norm().item() / (wb.norm().item() + 1e-8)
        br_rel = (wr - wb).norm().item() / (wb.norm().item() + 1e-8)
        pr_rel = (wr - wp).norm().item() / (wp.norm().item() + 1e-8)
        bp_cos = torch.nn.functional.cosine_similarity(wb.reshape(1,-1), wp.reshape(1,-1)).item()
        br_cos = torch.nn.functional.cosine_similarity(wb.reshape(1,-1), wr.reshape(1,-1)).item()
        
        print(f"  {k}")
        print(f"    Base→Prod: rel_diff={bp_rel:.4f}  cos={bp_cos:.4f}")
        print(f"    Base→Retr: rel_diff={br_rel:.4f}  cos={br_cos:.4f}")
        print(f"    Prod→Retr: rel_diff={pr_rel:.4f}")

# ── Summary: which model changed more from base? ──
print("\n" + "=" * 80)
print("FINAL ANALYSIS: Training trajectory comparison")
print("=" * 80)

total_bp = sum((sd_p[k].float() - sd_b[k].float()).norm().item()**2 
               for k in common if sd_b[k].shape == sd_p[k].shape) ** 0.5
total_br = sum((sd_r[k].float() - sd_b[k].float()).norm().item()**2 
               for k in common if sd_b[k].shape == sd_r[k].shape) ** 0.5
total_pr = sum((sd_r[k].float() - sd_p[k].float()).norm().item()**2 
               for k in common if sd_p[k].shape == sd_r[k].shape) ** 0.5

print(f"\n  Total L2 distance Base→Production:  {total_bp:.2f}")
print(f"  Total L2 distance Base→Retrained:   {total_br:.2f}")
print(f"  Total L2 distance Prod→Retrained:   {total_pr:.2f}")
print(f"\n  Ratio (Retr/Prod from base): {total_br/total_bp:.4f}")
print(f"  If ratio > 1: retrained changed MORE from base")
print(f"  If ratio < 1: retrained changed LESS from base")

print("\nDone!")
