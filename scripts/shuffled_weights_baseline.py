"""Shuffled-weights baseline: superellipse exponent on a randomized transformer.

Control for the contrastive-direction result. We destroy the trained
computation by permuting the entries of every weight matrix -- a
permutation WITHIN each matrix, so every matrix keeps its exact entry
distribution (and thus per-matrix norm/scale) but loses all learned
structure. We then run the standard pipeline on the shuffled model:

  1. re-extract the paper's contrastive DoM directions (same prompt
     sets, same layer) from the shuffled model's activations,
  2. compute FineWeb anchor activations on the shuffled model,
  3. run the canonical 2D iso-response sweep per pair,
  4. write sweep PKLs with a `_shufseed<k>` variant suffix.

If p > 2 reflects trained feature structure, the shuffled model should
give p ~ 2 (or no stable fit). Repeated over several permutation seeds
(default 5), each with fresh directions/anchors/sweeps.

All caches carry the shufseed tag, so nothing collides with (or reuses)
the canonical run's direction/activation/sweep caches. Only the FineWeb
token-id cache is shared with the canonical run: the tokenizer is
untouched by the weight shuffle, so the anchor token ids are identical.

Usage (GPU; ~40 s/pair on H100, so tier1 x 5 seeds ~ 1.5 h):
  python scripts/shuffled_weights_baseline.py --target gemma --layer 2

Then fit each seed with the canonical per-pair-threshold protocol:
  for k in 0 1 2 3 4; do
      python scripts/fit_pairs.py --target gemma --layer 2 \
          --variant_suffix _shufseed$k --per_pair_threshold --exact
  done

Notes:
  - Default pair set is tier1 (20 pairs). `--pair_set all` runs the full
    C(33, 2) = 528 pairs per seed; if you do that, apply the same
    intra-pair overlap prefilter downstream as for the canonical run.
  - `--scope blocks` (default) shuffles the 2D weight matrices of the
    transformer blocks only (attention + MLP), leaving embeddings and
    norms intact so the input representation stays defined. `--scope all`
    additionally shuffles every other 2D parameter (embedding matrix
    included; with tied embeddings that also randomizes the unembedding).
"""
from __future__ import annotations
import os, sys, pickle, argparse, time
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F

from src.model import load_model, _get_blocks
from scripts.lib import registry, activations as actlib
from scripts.lib.directions import compute_dom, _hash_list
from scripts.lib.sweep_core import run_2d_per_anchors
from scripts.lib.parametrize import eps_thresholds, CHART

SWEEP_DIR = "results/sweeps_2d"
DIRS_DIR = "results/directions"
os.makedirs(SWEEP_DIR, exist_ok=True)
os.makedirs(DIRS_DIR, exist_ok=True)

# canonical E2 sweep params (match sweep_2d.py)
N_STEPS_2D = 31
MAX_ANGLE_2D = 60.0
N_ANCHORS = 30


def ts(): return time.strftime("%H:%M:%S")


def shuffle_weights(model, seed: int, scope: str = "blocks"):
    """Permute the entries of every 2D weight matrix, within each matrix.

    scope='blocks': 2D parameters of the transformer blocks only.
    scope='all':    every 2D parameter of the model (embeddings included).

    Deterministic given (seed, scope): parameters are visited in sorted
    name order and drawn from one CPU generator seeded with `seed`.

    Returns (n_matrices, n_entries) shuffled.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if scope == "blocks":
        named = [(f"block{i}.{n}", p)
                 for i, blk in enumerate(_get_blocks(model))
                 for n, p in blk.named_parameters()]
    elif scope == "all":
        named = list(model.named_parameters())
    else:
        raise ValueError(f"unknown scope {scope}")
    named = sorted(((n, p) for n, p in named if p.ndim == 2),
                   key=lambda t: t[0])
    if not named:
        raise RuntimeError("no 2D weight matrices found to shuffle")
    n_entries = 0
    with torch.no_grad():
        for name, p in named:
            perm = torch.randperm(p.numel(), generator=g).to(p.device)
            p.data.copy_(p.data.reshape(-1)[perm].reshape(p.shape))
            n_entries += p.numel()
    return len(named), n_entries


def extract_dirs_shuffled(model, tokenizer, device, target, layer, names, tag):
    """DoM extraction on the shuffled model. Same protocol as
    scripts/lib/directions.extract_all, but cached under the shufseed tag
    so the canonical `dirs_<target>_L<layer>.pkl` is never touched."""
    pkl = os.path.join(DIRS_DIR, f"dirs_{target}_L{layer}_{tag}.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            blob = pickle.load(f)
        if all(n in blob["directions"] for n in names):
            print(f"[{ts()}] directions: reusing {pkl}", flush=True)
            return blob
        dirs, signs, hashes = (blob["directions"], blob["signs"],
                               blob.get("prompt_set_hashes", {}))
    else:
        dirs, signs, hashes = {}, {}, {}
    todo = [n for n in names if n not in dirs]
    print(f"[{ts()}] directions ({tag}): computing {len(todo)}", flush=True)
    for n in todo:
        a, b = registry.get_prompts(n)
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        t0 = time.time()
        d = compute_dom(model, tokenizer, device, layer, a, b)
        if not torch.isfinite(d).all():
            raise RuntimeError(f"non-finite DoM direction for {n} ({tag}); "
                               f"shuffled forward pass may have overflowed")
        dirs[n] = d
        signs[n] = registry.get_sign(n)
        hashes[n] = {"a_sha": _hash_list(a), "b_sha": _hash_list(b),
                     "n_pairs": m, "family": registry.family(n)}
        print(f"  {n:14s}  {time.time()-t0:.1f}s", flush=True)
    out = {"directions": dirs, "signs": signs, "layer": layer,
           "model": target, "weight_shuffle_tag": tag,
           "prompt_set_hashes": hashes}
    tmp = pkl + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, pkl)
    return out


def fineweb_acts_shuffled(model, tokenizer, device, target, layer, tag):
    """FineWeb anchors forwarded through the SHUFFLED model. Activations
    are cached as acts_<target>_<tag>_L<layer>_fineweb.pkl; the token-id
    cache is shared with the canonical run (same tokenizer, same seed)."""
    token_cache = f"{actlib.CACHE_ROOT}/fineweb_cache_{target}"
    os.makedirs(token_cache, exist_ok=True)
    from src.data import load_fineweb_fixed_length

    def token_loader():
        return load_fineweb_fixed_length(
            N_ANCHORS, tokenizer, seq_len=actlib.SEQ_LEN, seed=actlib.SEED,
            cache_dir=token_cache)

    return actlib._anchor_acts(model, tokenizer, device, f"{target}_{tag}",
                               layer, "fineweb", token_loader,
                               N_ANCHORS, actlib.SEQ_LEN, actlib.SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma",
                    choices=list(registry.TARGETS))
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma-separated permutation seeds")
    ap.add_argument("--scope", default="blocks", choices=["blocks", "all"],
                    help="which 2D weight matrices to shuffle (see module "
                         "docstring). Non-default scope appends _<scope> "
                         "to the tag.")
    ap.add_argument("--pair_set", default="tier1", choices=["tier1", "all"],
                    help="tier1 = the 20-pair v1 list (default; 528 pairs "
                         "per seed is expensive)")
    ap.add_argument("--pairs", default=None,
                    help="optional override, 'a,b' or 'a,b;c,d'")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    cfg = registry.TARGETS[args.target]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}

    if args.pairs:
        pair_list = [tuple(p.strip() for p in chunk.split(",")[:2])
                     for chunk in args.pairs.split(";")]
    elif args.pair_set == "tier1":
        pair_list = registry.TIER1_PAIRS
    else:
        pair_list = registry.all_pairs()
    needed = sorted({n for ab in pair_list for n in ab})

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; this will be extremely slow",
              flush=True)

    angles_deg = np.linspace(0, MAX_ANGLE_2D, N_STEPS_2D)
    angles_rad = np.deg2rad(angles_deg)

    for seed in seeds:
        tag = f"shufseed{seed}"
        if args.scope != "blocks":
            tag += f"_{args.scope}"
        suffix = f"_{tag}"

        todo = [(a, b) for (a, b) in pair_list
                if not (args.skip_existing and os.path.exists(os.path.join(
                    SWEEP_DIR,
                    f"sweep2d_{args.target}_L{args.layer}_{a}__{b}_fineweb_"
                    f"{int(MAX_ANGLE_2D)}deg{suffix}.pkl")))]
        if not todo:
            print(f"[{ts()}] seed {seed}: all {len(pair_list)} sweeps exist, "
                  f"skipping model load", flush=True)
            continue

        # Fresh load per seed: permutations must be independent, and a
        # reload is simpler and safer than undoing the previous shuffle.
        print(f"[{ts()}] === seed {seed}: loading {cfg['model']} ===",
              flush=True)
        model, tokenizer, device = load_model(cfg["model"],
                                              dtype=dtype_map[args.dtype])
        n_layers = len(_get_blocks(model))
        measure_layer = n_layers - 2   # canonical penultimate readout

        n_mat, n_ent = shuffle_weights(model, seed, scope=args.scope)
        print(f"[{ts()}] shuffled {n_mat} matrices, {n_ent/1e9:.2f}B entries "
              f"(scope={args.scope}, seed={seed})", flush=True)

        dirs_blob = extract_dirs_shuffled(
            model, tokenizer, device, args.target, args.layer, needed, tag)
        all_dirs = dirs_blob["directions"]
        signs = dirs_blob["signs"]

        print(f"[{ts()}] anchors (fineweb, through shuffled model)",
              flush=True)
        fw = fineweb_acts_shuffled(model, tokenizer, device, args.target,
                                   args.layer, tag)
        contexts, activations = fw["contexts"], fw["activations"]
        print(f"  N={len(activations)} "
              f"||a||={activations.norm(dim=-1).mean():.2f}", flush=True)

        for (a, b) in todo:
            out_path = os.path.join(
                SWEEP_DIR,
                f"sweep2d_{args.target}_L{args.layer}_{a}__{b}_fineweb_"
                f"{int(MAX_ANGLE_2D)}deg{suffix}.pkl")
            d1 = all_dirs[a].float()
            d2 = all_dirs[b].float()
            t0 = time.time()
            r = run_2d_per_anchors(model, contexts, activations, d1, d2,
                                   args.layer, measure_layer, device,
                                   angles_rad, mode="geodesic")
            print(f"  {a:>10s} x {b:<10s} {time.time()-t0:.1f}s "
                  f"l2_max={r['l2'].max():.2f}", flush=True)

            hashes_blob = dirs_blob.get("prompt_set_hashes", {})
            out = {
                "chart": CHART,
                "weight_shuffle": {"seed": seed, "scope": args.scope,
                                   "n_matrices": n_mat, "n_entries": n_ent},
                "prompt_set_hashes": {a: hashes_blob.get(a),
                                      b: hashes_blob.get(b),
                                      "source": None},
                "direction_family": "shuffled_dom",
                "seed_anchors": fw.get("seed", actlib.SEED),
                "seed_random_dirs": None,
                "direction_signs": {a: signs.get(a), b: signs.get(b)},
                "eps_thresholds": eps_thresholds(),
                "model": args.target,
                "model_full": cfg["model"],
                "perturb_layer": args.layer,
                "measure_layer": measure_layer,
                "measure_offset": -2,
                "perturb_pos": -1,
                "mode": "geodesic",
                "metrics": ["l2"],
                "direction_labels": (a, b),
                "source_name": "fineweb",
                "anchor_source": "fineweb",
                "angles_deg": angles_deg,
                "n_anchors": len(activations),
                "anchor_norm_mean": float(activations.norm(dim=-1).mean()),
                **{k: r[k] for k in r},
            }
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(out, f)
            os.replace(tmp, out_path)

        del model
        torch.cuda.empty_cache()

    print(f"[{ts()}] done. Fit with e.g.:", flush=True)
    for seed in seeds:
        tag = f"shufseed{seed}" + ("" if args.scope == "blocks"
                                   else f"_{args.scope}")
        print(f"  python scripts/fit_pairs.py --target {args.target} "
              f"--layer {args.layer} --variant_suffix _{tag} "
              f"--per_pair_threshold --exact", flush=True)


if __name__ == "__main__":
    main()
