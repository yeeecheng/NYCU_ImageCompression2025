#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VC_HW3 — Motion Estimation (ME) & Motion Compensation (MC)
Author: (fill your name)
Usage examples:
  Full search with multiple ranges:
    python vc_hw3_me_mc.py --ref one_gray.png --target two_gray.png --algo full --search 8 16 32
  Three-step search with R=8:
    python vc_hw3_me_mc.py --ref one_gray.png --target two_gray.png --algo tss --search 8
Outputs:
  - results_[algo].csv : PSNR & runtime per setting
  - reconstructed_[algo]_R{R}.png and residual_[algo]_R{R}.png
"""
import argparse, time, math, os
import numpy as np
from PIL import Image

BLOCK = 8

# -------------------------- Utils --------------------------
def load_gray(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.uint8)

def save_gray(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)

def psnr(x, y, maxv=255.0):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(maxv) - 10 * math.log10(mse)

def pad_for_search(ref, pad):
    return np.pad(ref, ((pad,pad),(pad,pad)), mode='edge')

def blocks_shape(h, w, bs=BLOCK):
    return (h // bs) * bs, (w // bs) * bs

# Sum of Absolute Differences for an 8x8 block
def sad(block_a, block_b):
    return np.sum(np.abs(block_a.astype(np.int32) - block_b.astype(np.int32)))

# --------------------- Full Search (FS) ---------------------
def full_search_me(ref, target, R, bs=BLOCK):
    """
    Returns motion vectors (dy, dx) per block in target relative to ref.
    Search window: center at the same block position, range [-R, R] (integer pixel).
    """
    h, w = target.shape
    H, W = blocks_shape(h, w, bs)
    ref = ref[:H, :W]
    target = target[:H, :W]
    ref_pad = pad_for_search(ref, R)

    mv = np.zeros((H//bs, W//bs, 2), dtype=np.int16)  # (dy, dx)
    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            t_blk = target[by:by+bs, bx:bx+bs]
            best_cost = 1e18
            best = (0, 0)
            # position in padded ref where "same" block would start
            r0 = by + R
            c0 = bx + R
            for dy in range(-R, R+1):
                for dx in range(-R, R+1):
                    r = r0 + dy
                    c = c0 + dx
                    r_blk = ref_pad[r:r+bs, c:c+bs]
                    cost = sad(t_blk, r_blk)
                    if cost < best_cost:
                        best_cost = cost
                        best = (dy, dx)
            mv[by//bs, bx//bs] = best
    return mv, ref[:H,:W], target[:H,:W]

# --------------- Three-Step Search (TSS) --------------------
def three_step_search_me(ref, target, R, bs=BLOCK):
    """
    Classic Three-Step Search (TSS) with integer precision.
    """
    h, w = target.shape
    H, W = blocks_shape(h, w, bs)
    ref = ref[:H, :W]
    target = target[:H, :W]
    ref_pad = pad_for_search(ref, R)

    def clip_center(y, x):
        y = max(0, min(H-bs, y))
        x = max(0, min(W-bs, x))
        return y, x

    mv = np.zeros((H//bs, W//bs, 2), dtype=np.int16)
    # initial step is highest power of two <= R, then halve each iteration
    step = 1
    while step*2 <= R: step *= 2

    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            t_blk = target[by:by+bs, bx:bx+bs]
            cy, cx = by, bx  # center in unpadded coords
            best = (0, 0)
            best_cost = 1e18
            s = step
            # search iteratively
            offy, offx = 0, 0
            while s >= 1:
                candidates = [(0,0), (-s,0), (s,0), (0,-s), (0,s), (-s,-s), (-s,s), (s,-s), (s,s)]
                # evaluate around current offset
                local_best = best
                local_best_cost = best_cost
                for dy, dx in candidates:
                    vy = offy + dy
                    vx = offx + dx
                    if abs(vy) > R or abs(vx) > R:
                        continue
                    r = cy + vy + R
                    c = cx + vx + R
                    r_blk = ref_pad[r:r+bs, c:c+bs]
                    cost = sad(t_blk, r_blk)
                    if cost < local_best_cost:
                        local_best_cost = cost
                        local_best = (vy, vx)
                offy, offx = local_best
                best_cost = local_best_cost
                s //= 2
            mv[by//bs, bx//bs] = (offy, offx)
    return mv, ref[:H,:W], target[:H,:W]

# ---------------- Reconstruction & Residual -----------------
def reconstruct_from_mv(ref, mv, bs=BLOCK):
    H, W = ref.shape
    rec = np.zeros_like(ref)
    R = max(abs(mv[...,0]).max(), abs(mv[...,1]).max())
    ref_pad = pad_for_search(ref, int(R))
    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            dy, dx = mv[by//bs, bx//bs]
            r = by + int(R) + dy
            c = bx + int(R) + dx
            rec[by:by+bs, bx:bx+bs] = ref_pad[r:r+bs, c:c+bs]
    return rec

def run_once(ref, target, algo, R):
    t0 = time.perf_counter()
    if algo == 'full':
        mv, ref_c, tgt_c = full_search_me(ref, target, R)
    elif algo == 'tss':
        mv, ref_c, tgt_c = three_step_search_me(ref, target, R)
    else:
        raise ValueError('Unknown algo')
    rec = reconstruct_from_mv(ref_c, mv)
    residual = (tgt_c.astype(np.int16) - rec.astype(np.int16) + 128).astype(np.uint8)  # add 128 for visualization
    t1 = time.perf_counter()
    return {
        'psnr': psnr(tgt_c, rec),
        'runtime_sec': t1 - t0,
        'reconstructed': rec,
        'residual_vis': residual,
        'cropped_ref': ref_c,
        'cropped_target': tgt_c,
        'mv': mv
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='reference (previous) frame path, grayscale')
    ap.add_argument('--target', required=True, help='current frame path, grayscale')
    ap.add_argument('--algo', choices=['full','tss'], required=True)
    ap.add_argument('--search', type=int, nargs='+', required=True, help='search ranges R (e.g., 8 16 32)')
    ap.add_argument('--outdir', default='outputs', help='directory to save outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ref = load_gray(args.ref)
    target = load_gray(args.target)

    rows = []
    for R in args.search:
        res = run_once(ref, target, args.algo, R)
        rows.append({'algo': args.algo, 'R': R, 'PSNR(dB)': res['psnr'], 'Runtime(s)': res['runtime_sec']})
        save_gray(res['reconstructed'], os.path.join(args.outdir, f'reconstructed_{args.algo}_R{R}.png'))
        save_gray(res['residual_vis'], os.path.join(args.outdir, f'residual_{args.algo}_R{R}.png'))

    import csv
    csv_path = os.path.join(args.outdir, f'results_{args.algo}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['algo','R','PSNR(dB)','Runtime(s)'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[Saved] {csv_path}')
    for r in rows:
        print(f"algo={r['algo']} R={r['R']}: PSNR={r['PSNR(dB)']:.3f} dB, Runtime={r['Runtime(s)']:.3f} s")

if __name__ == '__main__':
    main()
