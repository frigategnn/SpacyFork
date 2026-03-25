"""
Extract the learned causal graph directly from SPACY's trained model parameters.

This script does NOT need G_pred.pt — it reconstructs the graph from
the TemporalAdjacencyMatrix logits saved during training.

Usage (run from the SPACY directory):
    python scripts/extract_causal_graph.py --run_dir outputs/2026-03-25/03-19-37

It searches for wandb checkpoint files or the last.ckpt saved by Lightning.
If no checkpoint exists, it can also re-instantiate the model from config
and extract the graph in-memory.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Variate names (match preprocessing output order) ─────────────────────
# The preprocessing script prints these at the end. Adjust if needed.
DEFAULT_VARIATE_NAMES = [
    'o3', 'PM10', 'PM2_5_DRY', 'SO2', 'NO', 'NO2', 'CO',
    'RH', 'T2',
    'CAD', 'HTN', 'DBM', 'TB', 'BA', 'COPD', 'DEM',
]


def reconstruct_graph_from_logits(logits, logits_lag):
    """
    Reconstruct the causal graph probabilities from TemporalAdjacencyMatrix logits.

    logits:     shape (3, n*(n-1)/2) — instantaneous graph (ThreeWayGraphDist)
    logits_lag: shape (2, lag, n, n) — lagged graph (Bernoulli)

    Returns: G of shape (lag+1, n, n) with edge probabilities in [0, 1].
    """
    lag = logits_lag.shape[1]

    # --- Instantaneous graph ---
    # logits has 3 categories per edge pair (i,j) where i < j:
    #   [0] = no edge, [1] = forward (i→j), [2] = reverse (j→i)
    probs_3way = F.softmax(logits, dim=0)  # (3, n*(n-1)/2)
    num_pairs = logits.shape[1]
    # Solve n from n*(n-1)/2 = num_pairs
    n = int((1 + (1 + 8 * num_pairs) ** 0.5) / 2)

    G_inst = torch.zeros(n, n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            G_inst[i, j] = probs_3way[1, idx].item()  # forward: i→j
            G_inst[j, i] = probs_3way[2, idx].item()  # reverse: j→i
            idx += 1

    # --- Lagged graph ---
    probs_lag = F.softmax(logits_lag, dim=0)[1]  # (lag, n, n)

    G = torch.cat([G_inst.unsqueeze(0), probs_lag], dim=0)
    return G


def find_and_load_graph(run_dir):
    """Search for the graph in various locations."""

    # 1. Direct G_pred.pt
    for pattern in ['G_pred.pt', '**/G_pred.pt']:
        files = glob.glob(os.path.join(run_dir, pattern), recursive=True)
        if files:
            G = torch.load(files[0], map_location='cpu')
            print(f"✓ Loaded G_pred.pt from {files[0]}")
            return G

    # 2. Checkpoint files
    ckpt_patterns = ['**/*.ckpt', '**/last.ckpt', '**/best.ckpt']
    for pattern in ckpt_patterns:
        files = glob.glob(os.path.join(run_dir, pattern), recursive=True)
        if files:
            return load_graph_from_checkpoint(files[0])

    # 3. wandb run files (wandb saves model checkpoints)
    wandb_dirs = glob.glob(os.path.join(run_dir, '**/wandb/**/files'), recursive=True)
    for wd in wandb_dirs:
        files = glob.glob(os.path.join(wd, '*.ckpt'))
        if files:
            return load_graph_from_checkpoint(files[0])

    # 4. Search the entire working tree for the most recent logs dir
    logs_dirs = glob.glob(os.path.join(run_dir, '**/logs/**'), recursive=True)
    for ld in logs_dirs:
        files = glob.glob(os.path.join(ld, 'G_pred.pt'))
        if files:
            G = torch.load(files[0], map_location='cpu')
            print(f"✓ Loaded G_pred.pt from {files[0]}")
            return G

    return None


def load_graph_from_checkpoint(ckpt_path):
    """Load the graph logits from a Lightning checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state.get('state_dict', state)

    logits = sd.get('model.temporal_graph_dist.logits')
    logits_lag = sd.get('model.temporal_graph_dist.logits_lag')

    if logits is not None and logits_lag is not None:
        G = reconstruct_graph_from_logits(logits, logits_lag)
        print(f"✓ Reconstructed graph from checkpoint logits, shape: {tuple(G.shape)}")
        return G
    else:
        print(f"✗ Checkpoint does not contain temporal_graph_dist logits")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract and visualize the causal graph learned by SPACY')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Path to SPACY output dir (searches for G_pred.pt or checkpoints)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Direct path to a .ckpt file')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Edge probability threshold (default 0.3)')
    parser.add_argument('--num_variates', type=int, default=9,
                        help='Number of variates in the data')
    parser.add_argument('--out_dir', type=str, default='causal_graph_output',
                        help='Where to save heatmaps and results')
    args = parser.parse_args()

    # ── Load the graph ────────────────────────────────────────────────────
    G = None
    if args.checkpoint:
        G = load_graph_from_checkpoint(args.checkpoint)
    elif args.run_dir:
        G = find_and_load_graph(args.run_dir)

    if G is None:
        print("\nERROR: Could not find the causal graph.")
        print("Options:")
        print("  1. Pass --checkpoint /path/to/model.ckpt")
        print("  2. Pass --run_dir that contains G_pred.pt or .ckpt files")
        print("  3. Run `find . -name '*.ckpt' -o -name 'G_pred.pt'` to locate files")
        sys.exit(1)

    num_nodes = G.shape[-1]
    num_variates = args.num_variates
    names = DEFAULT_VARIATE_NAMES[:num_variates]
    # Pad names if fewer than expected
    while len(names) < num_variates:
        names.append(f'V{len(names)}')

    nodes_per_var = num_nodes // num_variates

    print(f"\n{'='*60}")
    print(f"Causal Graph Summary")
    print(f"{'='*60}")
    print(f"  Latent nodes:       {num_nodes}")
    print(f"  Variates:           {num_variates}")
    print(f"  Nodes per variate:  {nodes_per_var}")
    print(f"  Lag:                {G.shape[0] - 1}")
    print(f"  Threshold:          {args.threshold}")

    # ── Aggregate to variate-level ────────────────────────────────────────
    G_var = torch.zeros(G.shape[0], num_variates, num_variates)
    for lag_idx in range(G.shape[0]):
        for i in range(num_variates):
            for j in range(num_variates):
                block = G[lag_idx,
                          i*nodes_per_var:(i+1)*nodes_per_var,
                          j*nodes_per_var:(j+1)*nodes_per_var]
                G_var[lag_idx, i, j] = block.mean()

    # ── Print edges ───────────────────────────────────────────────────────
    lag_labels = ['instantaneous'] + [f'lag-{l}' for l in range(1, G.shape[0])]
    print(f"\n{'='*60}")
    print(f"Discovered Causal Edges (prob >= {args.threshold})")
    print(f"{'='*60}")

    edges = []
    for lag_idx in range(G_var.shape[0]):
        for i in range(num_variates):
            for j in range(num_variates):
                if i == j:
                    continue
                prob = G_var[lag_idx, i, j].item()
                if prob >= args.threshold:
                    edges.append((lag_labels[lag_idx], names[i], names[j], prob))

    edges.sort(key=lambda x: -x[3])  # sort by probability descending
    for lag_label, cause, effect, prob in edges:
        print(f"  {cause:15s} → {effect:15s}  ({lag_label}, prob={prob:.3f})")
    if not edges:
        print("  No edges above threshold. Try --threshold 0.1")

    # ── Highlight pollution → health edges ────────────────────────────────
    print(f"\n{'='*60}")
    print("Pollution → Health edges (the key causal question):")
    print(f"{'='*60}")
    poll_health = [e for e in edges
                   if any(p in e[1] for p in ['o3','PM10','PM2_5','SO2','NO','NO2','CO'])
                   and any(h in e[2] for h in ['CAD','HTN','DBM','TB','BA','COPD','DEM'])]
    for lag_label, cause, effect, prob in poll_health:
        print(f"  {cause:15s} → {effect:15s}  ({lag_label}, prob={prob:.3f})")
    if not poll_health:
        print("  None found. Model may need more data/epochs or lower threshold.")

    # ── Save heatmaps ─────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    for lag_idx in range(G_var.shape[0]):
        fig, ax = plt.subplots(figsize=(10, 8))
        mat = G_var[lag_idx].numpy()
        sns.heatmap(mat, xticklabels=names, yticklabels=names,
                    annot=True, fmt='.2f', cmap='RdYlBu_r',
                    vmin=0, vmax=1, ax=ax,
                    linewidths=0.5, square=True)
        ax.set_title(f'Learned Causal Graph – {lag_labels[lag_idx]} edges')
        ax.set_xlabel('Effect ←')
        ax.set_ylabel('Cause →')
        plt.tight_layout()
        fname = os.path.join(args.out_dir, f'causal_graph_{lag_labels[lag_idx]}.png')
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"\nSaved: {fname}")

    torch.save(G_var, os.path.join(args.out_dir, 'G_variate_level.pt'))
    print(f"Saved: {args.out_dir}/G_variate_level.pt")

    print(f"\n{'='*60}")
    print("Interpretation:")
    print("  Rows = cause, Columns = effect")
    print("  G[0] = instantaneous (same time step)")
    print("  G[1:] = lagged (cause precedes effect)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
