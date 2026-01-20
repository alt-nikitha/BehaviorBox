
#!/usr/bin/env python3
"""


This code normalizes downstream JSON summaries into a long (tidy) table and
produces per-model plots. It supports aggregated metrics across splits (avg/max)
and produces one subplot per dataset subset (topic) for each model.

Usage examples:
  python analysis/plot_downstream_performance.py \
    --results-dir /home/nsrikant/BehaviorBoxNew/results/mmlu \
    --out-dir /home/nsrikant/BehaviorBoxNew/analysis/plots \
    --agg avg

Arguments:
  --agg: 'avg' (default), 'max', or 'both' to control aggregation across test/dev/validation.
  --pattern: filename glob (default uses dataset name to build `<dataset>_summary*.json`).

Produces:
 - CSV aggregated long table: analysis/plots/aggregated_<dataset>_long.csv
 - Per-model overall plot: analysis/plots/overall_<model>.png
 - Per-model subset plots: analysis/plots/subsets_<model>[_avg|max].png

Dependencies: pandas, matplotlib, seaborn
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_json_files(results_dir: Path, pattern: str) -> List[Path]:
    return list(results_dir.rglob(pattern))


def parse_model_and_checkpoint(rel_path: Path) -> Tuple[str, str]:
    parts = rel_path.parts
    if len(parts) == 0:
        return "unknown", ""
    model_dir = parts[0]
    m = re.match(r"(?P<size>pythia-[^\-]+(?:_[0-9]+[a-z]?)?)(?:-(?P<ckpt>.*))?", model_dir)
    if m:
        return m.group('size'), (m.group('ckpt') or '')
    return model_dir, ""


def extract_split_scores(per_split_accuracy: dict) -> Dict[str, Dict[str, float]]:
    """Return mapping: topic -> {splitname: value, ...}.

    Handles keys like 'topic_test' or plain 'topic'.
    """
    by_topic = defaultdict(dict)
    for k, v in per_split_accuracy.items():
        if not isinstance(k, str):
            continue
        if '_' not in k:
            by_topic[k]['main'] = v
            continue
        base, split = k.rsplit('_', 1)
        by_topic[base][split] = v
    return dict(by_topic)


def checkpoint_sort_key(ckpt_label: str) -> int:
    if not ckpt_label:
        return -1
    m = re.search(r"(\d+)", ckpt_label)
    if m:
        return int(m.group(1))
    return 10 ** 9


def extract_ckpt_num(ckpt_label: str):
    if not ckpt_label:
        return None
    m = re.search(r"(\d+)", ckpt_label)
    return int(m.group(1)) if m else None


def aggregate_long(results_dir: Path, pattern: str, agg: str = 'avg') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Read JSON summaries and return a long DataFrame with columns:
    model_size, checkpoint, topic, split, value, overall_accuracy, metric, total_pairs

    If explicit splits are present for a topic (e.g., test/dev/validation), this
    creates rows for each split and also aggregated rows (split='avg' and split='max')
    depending on the agg argument. If no explicit splits are present, creates a
    'main' split row and duplicates it into avg/max so downstream plotting is
    uniform.
    """
    files = [file for file in find_json_files(results_dir, pattern) if "olmo" not in file.name.lower()]
    rows: List[Dict[str, Any]] = []
    topics = set()
    splits = set()

    for p in files:
        try:
            data = json.loads(p.read_text())
        except Exception:
            print(f"Warning: failed to read/parse {p}")
            continue
        rel = p.relative_to(results_dir)
        model_size, ckpt = parse_model_and_checkpoint(rel)
        overall = data.get('overall_accuracy')
        metric = data.get('metric')
        total_pairs = data.get('total_pairs')
        per = data.get('per_split_accuracy', {})
        topic_map = extract_split_scores(per)
        has_splits = any(len(s) > 1 or (list(s.keys()) != ['main']) for s in topic_map.values())

        for topic, splitvals in topic_map.items():
            # splitvals is {splitname: val, ...} where splitname might be 'main' or 'test' etc.
            if has_splits:
                # per-split rows
                vals = []
                for splitname, v in splitvals.items():
                    splits.add(splitname)
                    rows.append({
                        'model_size': model_size,
                        'checkpoint': ckpt,
                        'topic': topic,
                        'split': splitname,
                        'value': v,
                        'overall_accuracy': overall,
                        'metric': metric,
                        'total_pairs': total_pairs,
                        'file': str(p),
                    })
                    topics.add(topic)
                    vals.append(v)
                # aggregated rows
                if vals:
                    avgv = sum(vals) / len(vals)
                    maxv = max(vals)
                else:
                    avgv = None
                    maxv = None
                rows.append({
                    'model_size': model_size,
                    'checkpoint': ckpt,
                    'topic': topic,
                    'split': 'avg',
                    'value': avgv,
                    'overall_accuracy': overall,
                    'metric': metric,
                    'total_pairs': total_pairs,
                    'file': str(p),
                })
                rows.append({
                    'model_size': model_size,
                    'checkpoint': ckpt,
                    'topic': topic,
                    'split': 'max',
                    'value': maxv,
                    'overall_accuracy': overall,
                    'metric': metric,
                    'total_pairs': total_pairs,
                    'file': str(p),
                })
            else:
                # no explicit splits: treat the single value as 'main' and duplicate to avg/max
                v = list(splitvals.values())[0] if splitvals else None
                rows.append({
                    'model_size': model_size,
                    'checkpoint': ckpt,
                    'topic': topic,
                    'split': 'main',
                    'value': v,
                    'overall_accuracy': overall,
                    'metric': metric,
                    'total_pairs': total_pairs,
                    'file': str(p),
                })
                # make avg/max same as main for uniform handling
                rows.append({
                    'model_size': model_size,
                    'checkpoint': ckpt,
                    'topic': topic,
                    'split': 'avg',
                    'value': v,
                    'overall_accuracy': overall,
                    'metric': metric,
                    'total_pairs': total_pairs,
                    'file': str(p),
                })
                rows.append({
                    'model_size': model_size,
                    'checkpoint': ckpt,
                    'topic': topic,
                    'split': 'max',
                    'value': v,
                    'overall_accuracy': overall,
                    'metric': metric,
                    'total_pairs': total_pairs,
                    'file': str(p),
                })

    if not rows:
        return pd.DataFrame(), [], []

    df = pd.DataFrame(rows)

    # compute ckpt_pos per model by ordering unique checkpoints using checkpoint_sort_key
    ckpt_pos_map: Dict[Tuple[str, str], int] = {}
    for model in df['model_size'].unique():
        ckpts = list(df[df['model_size'] == model]['checkpoint'].unique())
        ckpts_sorted = sorted(ckpts, key=lambda c: checkpoint_sort_key(c))
        for i, c in enumerate(ckpts_sorted):
            ckpt_pos_map[(model, c)] = i
    df['ckpt_pos'] = df.apply(lambda r: ckpt_pos_map.get((r['model_size'], r['checkpoint']), 0), axis=1)

    return df, sorted(topics), sorted(splits)


def plot_overall(df_long: pd.DataFrame, out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_df = df_long[['model_size', 'checkpoint', 'ckpt_pos', 'overall_accuracy']].drop_duplicates()
    for model, g in overall_df.groupby('model_size'):
        gg = g.sort_values('ckpt_pos')
        plt.figure(figsize=(8, 4))
        sns.lineplot(x='ckpt_pos', y='overall_accuracy', marker='o', data=gg)
        plt.xticks(gg['ckpt_pos'], gg['checkpoint'], rotation=45)
        plt.xlabel('Checkpoint (ordinal)')
        plt.ylabel('Overall accuracy (%)')
        plt.title(f'Overall accuracy across checkpoints — {model} ({dataset})')
        plt.tight_layout()
        fn = out_dir / f'overall_{dataset}_{model}.png'
        plt.savefig(fn)
        plt.close()
        print(f"Saved {fn}")


def plot_subsets_long(df_long: pd.DataFrame, out_dir: Path, agg_modes: List[str], max_topics: int | None = None, dataset: str = ''):
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = sorted(df_long['topic'].unique())
    for agg in agg_modes:
        sel = df_long[df_long['split'] == agg]
        if sel.empty:
            print(f"No data for aggregation '{agg}' — skipping")
            continue
        for model, g in sel.groupby('model_size'):
            model_topics = sorted(g['topic'].unique())
            if max_topics:
                model_topics = model_topics[:max_topics]
            if not model_topics:
                continue
            n = len(model_topics)
            cols = 4
            rows = (n + cols - 1) // cols
            figsize = (cols * 4, rows * 3)
            fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes_flat = axes.flatten()
            mapping = {row['ckpt_pos']: row['checkpoint'] for _, row in g.drop_duplicates('ckpt_pos').iterrows()}

            for i, topic in enumerate(model_topics):
                ax = axes_flat[i]
                df_topic = g[g['topic'] == topic].sort_values('ckpt_pos')
                if df_topic['value'].isnull().all():
                    ax.set_visible(False)
                    continue
                # plot a single line across ckpt positions for this topic
                ax.plot(df_topic['ckpt_pos'], df_topic['value'], marker='o')
                ax.set_title(topic)
                xticks = df_topic['ckpt_pos'].unique()
                ax.set_xticks(xticks)
                ax.set_xticklabels([mapping.get(x, str(x)) for x in xticks], rotation=45, fontsize=8)
                ax.set_ylabel('Accuracy (%)')

            for j in range(len(model_topics), len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(f"Per-topic ({len(model_topics)}) — agg={agg} — model {model} ({dataset})")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fn = out_dir / f'subsets_{dataset}_{model}_{agg}.png'
            fig.savefig(fn)
            plt.close(fig)
            print(f"Saved {fn}")


def plot_subsets_overlay(df_long: pd.DataFrame, out_dir: Path, agg_modes: List[str], dataset: str = '', max_topics: int | None = None):
    """Produce one overlay plot per model where each checkpoint is a separate colored line.

    X-axis: topic (categorical). Each line: a checkpoint (ordered by ckpt_pos).
    This is useful when you want to compare checkpoint behavior across topics in a single figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for agg in agg_modes:
        sel = df_long[df_long['split'] == agg]
        if sel.empty:
            print(f"No data for aggregation '{agg}' — skipping overlay plots")
            continue
        for model, g in sel.groupby('model_size'):
            model_topics = sorted(g['topic'].unique())
            if max_topics:
                model_topics = model_topics[:max_topics]
            if not model_topics:
                continue

            # pivot to have topics on x-axis and one series per checkpoint
            pivot = g.pivot_table(index=['topic'], columns=['ckpt_pos', 'checkpoint'], values='value')
            if pivot.empty:
                continue

            # ensure topics are ordered as in model_topics
            pivot = pivot.reindex(model_topics)

            plt.figure(figsize=(max(8, len(model_topics) * 0.6), 6))
            # for each checkpoint (multiindex columns), plot its values across topics
            for ckpt_pos, checkpoint in pivot.columns:
                series = pivot[(ckpt_pos, checkpoint)]
                if series.isnull().all():
                    continue
                # use ckpt_pos as an ordinal in the legend to keep order
                plt.plot(range(len(series.index)), series.values, marker='o', label=f"{checkpoint}")

            plt.xticks(range(len(model_topics)), model_topics, rotation=45, fontsize=8)
            plt.xlabel('Topic')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Checkpoint overlay across topics — agg={agg} — model {model} ({dataset})')
            plt.legend(title='checkpoint', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            fn = out_dir / f'subsets_overlay_{dataset}_{model}_{agg}.png'
            plt.savefig(fn, bbox_inches='tight')
            plt.close()
            print(f"Saved {fn}")


def group_models_by_ckpts(df: pd.DataFrame) -> Dict[Tuple[str, ...], List[str]]:
    """Return mapping from tuple(sorted_checkpoint_labels) -> list of model_size that have exactly that checkpoint sequence.

    The checkpoint sequence is ordered using checkpoint_sort_key to have a canonical ordering.
    """
    model_ckpts: Dict[str, Tuple[str, ...]] = {}
    for model in df['model_size'].unique():
        ckpts = list(df[df['model_size'] == model]['checkpoint'].dropna().unique())
        # sort by numeric where possible to get canonical sequence
        ckpts_sorted = tuple(sorted(ckpts, key=lambda c: checkpoint_sort_key(c)))
        model_ckpts[model] = ckpts_sorted
    groups: Dict[Tuple[str, ...], List[str]] = {}
    for model, ckpts in model_ckpts.items():
        groups.setdefault(ckpts, []).append(model)
    return groups


def plot_overall_grouped_by_ckpts(df_long: pd.DataFrame, out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = group_models_by_ckpts(df_long)
    for ckpt_tuple, models in groups.items():
        if not ckpt_tuple:
            # skip models without checkpoints
            continue
        plt.figure(figsize=(8, 4))
        # x positions for this shared ckpt sequence
        x = list(range(len(ckpt_tuple)))
        for model in models:
            sub = df_long[(df_long['model_size'] == model)][['checkpoint', 'ckpt_pos', 'overall_accuracy']].drop_duplicates()
            # map checkpoint -> value
            mapping = {r['checkpoint']: r['overall_accuracy'] for _, r in sub.iterrows()}
            y = [mapping.get(ck, None) for ck in ckpt_tuple]
            plt.plot(x, y, marker='o', label=model)
        plt.xticks(x, ckpt_tuple, rotation=45)
        plt.xlabel('Checkpoint')
        plt.ylabel('Overall accuracy (%)')
        title = f"Overall accuracy — models {', '.join(models)} ({dataset})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        fn = out_dir / f'overall_{dataset}_{"__".join(models)}.png'
        plt.savefig(fn)
        plt.close()
        print(f"Saved {fn}")


def plot_subsets_grouped_by_ckpts(df_long: pd.DataFrame, out_dir: Path, agg_modes: List[str], dataset: str = '', max_topics: int | None = None):
    """For each group of models that share the same checkpoint sequence, produce per-topic subplots overlaying the models.

    For each group and aggregation, create a grid of topic subplots where each model in the group is a separate line across the shared checkpoint positions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = group_models_by_ckpts(df_long)
    for agg in agg_modes:
        sel = df_long[df_long['split'] == agg]
        if sel.empty:
            print(f"No data for aggregation '{agg}' — skipping")
            continue
        for ckpt_tuple, models in groups.items():
            if not ckpt_tuple or len(models) < 2:
                # nothing to overlay (either no ckpts or only one model)
                continue
            # collect topics present across these models
            topics = sorted(set(sel[sel['model_size'].isin(models)]['topic'].unique()))
            if max_topics:
                topics = topics[:max_topics]
            if not topics:
                continue

            n = len(topics)
            cols = 4
            rows = (n + cols - 1) // cols
            figsize = (cols * 4, rows * 3)
            fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes_flat = axes.flatten()

            for i, topic in enumerate(topics):
                ax = axes_flat[i]
                for model in models:
                    # build mapping checkpoint -> value for this model/topic
                    df_mt = sel[(sel['model_size'] == model) & (sel['topic'] == topic)].drop_duplicates(subset=['checkpoint'])
                    if df_mt.empty:
                        continue
                    val_map = {r['checkpoint']: r['value'] for _, r in df_mt.iterrows()}
                    y = [val_map.get(ck, None) for ck in ckpt_tuple]
                    # plot, skipping if all None
                    if all(v is None for v in y):
                        continue
                    ax.plot(range(len(ckpt_tuple)), y, marker='o', label=model)

                ax.set_title(topic)
                ax.set_xticks(range(len(ckpt_tuple)))
                ax.set_xticklabels(ckpt_tuple, rotation=45, fontsize=8)
                ax.set_ylabel('Accuracy (%)')
                ax.legend(fontsize=8)

            for j in range(len(topics), len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(f"Per-topic overlay — agg={agg} — models {', '.join(models)} ({dataset})")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fn = out_dir / f'subsets_grouped_{dataset}_{"__".join(models)}_{agg}.png'
            fig.savefig(fn, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {fn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='blimp_full')
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results' / 'blimp_full')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[0] / 'plots')
    parser.add_argument('--pattern', type=str, default=None, help='Filename glob for JSON summaries (overrides dataset default)')
    parser.add_argument('--agg', type=str, default='avg', choices=['avg', 'max', 'both'], help='Aggregation across splits: avg, max, or both')
    parser.add_argument('--max-subsets', type=int, default=None)
    parser.add_argument('--overlay-checkpoints', action='store_true', help='Create per-model overlay plots where each checkpoint is a colored line across topics')
    parser.add_argument('--overlay-same-checkpoints', action='store_true', help='Overlay models that share the same checkpoint set (e.g., pythia-160m with pythia-6.9b) on the same plot')
    args = parser.parse_args()

    import sys

    results_dir = args.results_dir
    out_dir = args.out_dir
    # If the user provided a pattern, use it. Otherwise infer pattern from either
    # an explicitly passed --dataset or from the results directory name. This
    # prevents the script from defaulting to 'mmlu' when the user points
    # --results-dir at e.g. results/blimp_full but doesn't also pass --dataset.
    if args.pattern:
        pattern = args.pattern
        dataset = args.dataset
    else:
        # prefer an explicit --dataset on the command line if present
        if '--dataset' in sys.argv:
            dataset = args.dataset
        else:
            dataset = results_dir.name
        pattern = f"{dataset}_summary*.json"
    # create a subdirectory per dataset to avoid overwriting plots when running
    # different datasets into the same out-dir
    dataset_out = out_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)

    print(f"Reading results from {results_dir} pattern={pattern}")
    df_long, topics, splits = aggregate_long(results_dir, pattern, agg=args.agg)
    if df_long.empty:
        print("No result files found. Exiting.")
        return

    csv_path = dataset_out / f'aggregated_{dataset}_long.csv'
    df_long.to_csv(csv_path, index=False)
    print(f"Wrote long aggregated CSV to {csv_path}")

    # overall
    plot_overall(df_long, dataset_out, dataset)
    # if requested, also produce overall overlays for groups that share checkpoints
    if args.overlay_same_checkpoints:
        plot_overall_grouped_by_ckpts(df_long, dataset_out, dataset)

    # subsets: determine agg modes to plot
    agg_modes = ['avg'] if args.agg == 'avg' else (['max'] if args.agg == 'max' else ['avg', 'max'])
    if args.overlay_same_checkpoints:
        plot_subsets_grouped_by_ckpts(df_long, dataset_out, agg_modes, dataset=dataset, max_topics=args.max_subsets)
    elif args.overlay_checkpoints:
        plot_subsets_overlay(df_long, dataset_out, agg_modes, dataset=dataset, max_topics=args.max_subsets)
    else:
        plot_subsets_long(df_long, dataset_out, agg_modes, max_topics=args.max_subsets, dataset=dataset)


if __name__ == '__main__':
    main()
