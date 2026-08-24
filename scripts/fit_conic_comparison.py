"""Superellipse vs tilted-ellipse fits on identical contour points.

Produces the table in the paper's "Ruling out quadratic responses with cross
terms" appendix: for every direction pair we extract the iso-response contour
exactly as in the main analysis and fit both one-parameter families,

    superellipse    |x|^p + |y|^p = 1            (parameter p)
    tilted ellipse  x^2 + 2*delta*x*y + y^2 = 1  (parameter delta)

then compare their radial fit residuals with a paired two-sided sign test.
`delta` is the paper's symbol for the normalised cross term; `rho` is reserved
for the radial fit residual.

Output:
  results/conic_comparison_<target>_L<layer>.csv
  also prints the paper table to stdout.

Two things this script must get right, both of which an earlier ad-hoc version
got wrong:

  * Every family is scoped to one target/layer. Globbing `*_60deg_dirrandom.pkl`
    without the `sweep2d_<target>_L<layer>_` prefix also matches the random
    sweeps of every other model and layer (664 files instead of 528).
  * Every family gets the same intra-pair `|cos| < max_overlap` filter that the
    beeswarms apply, so the pair sets match Figure 2 (contrastive 318,
    MELBO 507, SAE 395, PCA 528, random 528). Note the sweep filenames embed
    the anchor tag before the window, e.g.
    `sweep2d_gemma_L2_melbo_000__melbo_001_fineweb_60deg_dirmelbo.pkl`,
    so the pair names must be parsed with the full `_fineweb_60deg_<tag>.pkl`
    suffix or the overlap lookup silently misses and the filter no-ops.
"""
from __future__ import annotations
import os, sys, glob, csv, pickle, argparse
from math import comb
sys.path.insert(0, ".")
import numpy as np

from scripts.lib.superellipse import (extract_contour, axis_intercept,
                                      fit_superellipse)
from scripts.plotting.plot_beeswarm_direction_types import _filter_family_overlap
from scripts.plotting.plot_robustness_beeswarm import _filter_exclude

SWEEPS = "results/sweeps_2d"

# label -> (sweep-file suffix, direction-cache suffix)
FAMILIES = [
    ("Contrastive", "_fineweb_60deg.pkl",                    None),
    ("MELBO",       "_fineweb_60deg_dirmelbo.pkl",           "_melbo"),
    ("SAE",         "_fineweb_60deg_dirsae_fineweb.pkl",     "_sae_fineweb"),
    ("PCA",         "_fineweb_60deg_dirpca_fineweb.pkl",     "_pca_fineweb"),
    ("Random",      "_fineweb_60deg_dirrandom.pkl",          "_random"),
    ("Random-diff", "_fineweb_60deg_dirrandomdiffavg_fineweb.pkl", "_randomdiffavg_fineweb"),
]


def normalized_points(pkl_path):
    """Iso-response contour in the per-axis normalised coordinates of the
    superellipse equation. Returns (x, y) or None if the pair does not fit."""
    d = pickle.load(open(pkl_path, "rb"))
    ang, grid = d["angles_deg"], np.median(d["l2"], axis=0)
    thr = 0.5 * min(grid[:, 0].max(), grid[0, :].max())
    raw = extract_contour(ang, grid, thr)
    if raw.size == 0:
        return None
    gm = float(ang.max())
    raw = raw[(raw[:, 0] < gm - 1.0) & (raw[:, 1] < gm - 1.0)]
    if len(raw) < 3:
        return None
    t1 = axis_intercept(ang, grid[:, 0], thr)
    t2 = axis_intercept(ang, grid[0, :], thr)
    if not (np.isfinite(t1) and np.isfinite(t2)) or t1 <= 0 or t2 <= 0:
        return None
    r_deg = np.hypot(raw[:, 0], raw[:, 1])
    r_safe = np.where(r_deg > 1e-12, r_deg, 1.0)
    r_rad = np.deg2rad(r_deg)
    xn = (np.sin(r_rad) / np.sin(np.deg2rad(t1))) * (raw[:, 0] / r_safe)
    yn = (np.sin(r_rad) / np.sin(np.deg2rad(t2))) * (raw[:, 1] / r_safe)
    m = (xn > 0.05) & (yn > 0.05) & (xn < 1.5) & (yn < 1.5)
    xn, yn = xn[m], yn[m]
    return (xn, yn) if len(xn) >= 3 else None


def fit_tilted_ellipse(x, y):
    """Least squares in delta for x^2 + 2*delta*x*y + y^2 = 1 (linear in delta).
    Returns (delta, mean radial fit residual)."""
    a, b = x * x + y * y - 1.0, 2.0 * x * y
    delta = -float(np.sum(a * b) / np.sum(b * b))
    v = np.clip(x * x + 2 * delta * x * y + y * y, 1e-9, None)
    return delta, float(np.abs(np.sqrt(v) - 1.0).mean())


def sign_test(wins, n):
    k = min(wins, n - wins)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n, 1.0)


def family_files(target, layer, max_overlap, pos_csv, exclude):
    """One sweep pkl per surviving pair, per family, scoped to target/layer."""
    prefix = f"sweep2d_{target}_L{layer}_"

    def pair_of(path, suffix):
        stem = os.path.basename(path)[len(prefix):-len(suffix)]
        return tuple(stem.split("__", 1)) if "__" in stem else None

    keep = set()
    if os.path.exists(pos_csv):
        for row in csv.DictReader(open(pos_csv)):
            keep.add((row["a"], row["b"]))
            keep.add((row["b"], row["a"]))

    out = {}
    for label, suffix, dir_suffix in FAMILIES:
        files = sorted(glob.glob(f"{SWEEPS}/{prefix}*{suffix}"))
        pairs = {}
        for p in files:
            pr = pair_of(p, suffix)
            if pr is None:
                continue
            # Contrastive is selected by the curated positive-pair list; the
            # other families carry a direction cache and use the |cos| filter.
            if dir_suffix is None and pr not in keep:
                continue
            pairs[frozenset(pr)] = p
        if dir_suffix is not None:
            kept = _filter_family_overlap({k: 0.0 for k in pairs}, target, layer,
                                          dir_suffix, max_overlap)
            pairs = {k: v for k, v in pairs.items() if k in kept}
        pairs = _filter_exclude(pairs, exclude)
        out[label] = sorted(pairs.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gemma")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--max_overlap", type=float, default=0.10)
    ap.add_argument("--exclude_dirs", default="Formal,HonestyShort,TensePresent",
                    help="Redundant contrastive directions pruned at plot time; "
                         "must match render_figures.sh.")
    ap.add_argument("--pos_csv", default="results/pos_pairs_gemma_L2.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    excl = {n.strip() for n in args.exclude_dirs.split(",") if n.strip()}
    fams = family_files(args.target, args.layer, args.max_overlap,
                        args.pos_csv, excl)

    rows, table = [], []
    for label, files in fams.items():
        rs = []
        for path in files:
            try:
                pts = normalized_points(path)
            except Exception:
                pts = None
            if pts is None:
                continue
            fs = fit_superellipse(*pts)
            if not np.isfinite(fs["p"]):
                continue
            delta, resid_delta = fit_tilted_ellipse(*pts)
            rs.append((fs["p"], fs["mean_radial_frac"], delta, resid_delta))
            rows.append(dict(family=label, file=os.path.basename(path),
                             p=fs["p"], resid_p=fs["mean_radial_frac"],
                             delta=delta, resid_delta=resid_delta))
        if not rs:
            print(f"{label}: no fits (check sweep filenames)", file=sys.stderr)
            continue
        P, rp, D, rd = map(np.array, zip(*rs))
        n, wins = len(rs), int((rd < rp).sum())
        table.append((label, n, np.median(P), np.median(D),
                      100 * np.median(rp), 100 * np.median(rd),
                      wins, sign_test(wins, n)))

    out = args.out or f"results/conic_comparison_{args.target}_L{args.layer}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'Family':<13}{'n':>5}{'med p':>8}{'med delta':>11}"
           f"{'resid p':>9}{'resid delta':>13}{'ellipse wins':>14}{'sign test':>12}")
    print(hdr)
    print("-" * len(hdr))
    for label, n, mp, md, rp, rd, wins, pv in table:
        print(f"{label:<13}{n:>5}{mp:>8.2f}{md:>11.3f}{rp:>8.2f}%{rd:>12.2f}%"
              f"{f'{wins}/{n}':>14}{pv:>12.1e}")
    print(f"\nper-pair fits written to {out}")


if __name__ == "__main__":
    main()
