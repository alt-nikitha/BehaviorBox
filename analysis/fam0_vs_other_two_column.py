"""Two-column HTML report: left = tokens the logreg confidently+correctly calls fam0 (sorted
by descending p(fam0)); right = tokens it confidently+correctly calls something else (sorted
by descending p(true_label), true_label != 0). Token + context for each, using the same saved
clf+scaler as fam0_vs_other_confident_vocab.py / logreg_weight_projection.py (no retraining).

Usage:
  python fam0_vs_other_two_column.py --model fam0_logreg_weights.pkl --top-n 100
"""
import argparse, html, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DIR = Path(__file__).resolve().parent


def render_html(rowsA, rowsB, out_path, top_n):
    esc = html.escape

    def render_col(rows):
        parts = []
        for rank, (p, fam, wid, e) in enumerate(rows, 1):
            parts.append(
                f'<div class="tok"><div class="meta">#{rank} p={p:.4f} '
                f'<span class="fam">{esc(FAM_LABELS.get(fam, str(fam)))}</span></div>'
                f'<div class="ctx"><span class="before">{esc(e["before"])}</span>'
                f'<span class="w">{esc(e["word"])}</span>'
                f'<span class="after">{esc(e["after"])}</span></div></div>'
            )
        return "\n".join(parts)

    doc = f"""<title>fam0 vs non-fam0: confidently correct tokens</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a}}
h1{{font-size:20px;margin:0 0 4px}} .meta-top{{color:#666;font-size:13px;margin-bottom:20px}}
.cols{{display:flex;gap:20px}}
.col{{flex:1;min-width:0}}
.col h2{{font-size:15px;margin:0 0 10px;position:sticky;top:0;background:#fafafa;padding:6px 0}}
.tok{{background:#fff;border:1px solid #e3e3e3;border-radius:6px;margin-bottom:8px;padding:8px 10px}}
.meta{{font-size:11.5px;color:#888;margin-bottom:4px;font-family:ui-monospace,Menlo,monospace}}
.fam{{color:#555}}
.ctx{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;white-space:pre-wrap;word-break:break-word}}
.before,.after{{color:#777}}
.w{{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}}
</style>
<h1>Confidently + correctly classified tokens: fam0 vs. not-fam0</h1>
<div class="meta-top">left: true label fam0, predicted fam0, sorted by p(fam0) desc (top {top_n}) &middot;
right: true label != fam0, predicted correctly, sorted by p(true label) desc (top {top_n})</div>
<div class="cols">
<div class="col"><h2>IS fam0 (late-reasoning/68k)</h2>
{render_col(rowsA)}
</div>
<div class="col"><h2>NOT fam0</h2>
{render_col(rowsB)}
</div>
</div>
"""
    out_path.write_text(doc)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--context-words", type=int, default=10)
    ap.add_argument("--out", default="fam0_vs_other_two_column.html")
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}")

    fam_names, F, members = build_families(0.45)
    R, D, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D, valid, sargs["min_r"], sargs["max_l1"],
                                     exclude=set(sargs["exclude_family"]))
    rows, labels_sel = sample_equal_exclusive(mask, labels, sargs["per_family"], sargs["seed"], fam_names)
    print(f"sampled {len(rows):,} tokens total (reproducing saved split); reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=sargs["seed"], stratify=labels_sel)
    X_te = scaler.transform(X[idx_te])
    y_te = labels_sel[idx_te]
    rows_te = rows[idx_te]

    proba = clf.predict_proba(X_te)
    pred = np.array(classes)[np.argmax(proba, axis=1)]

    col0 = classes.index(0)
    maskA = (y_te == 0) & (pred == 0)
    candA = np.where(maskA)[0]
    candA = candA[np.argsort(-proba[candA, col0])][: a.top_n]
    pA = proba[candA, col0]

    maskB = (y_te != 0) & (pred == y_te)
    candB = np.where(maskB)[0]
    conf_b = proba[candB, [classes.index(t) for t in y_te[candB]]]
    order_b = np.argsort(-conf_b)[: a.top_n]
    candB = candB[order_b]
    pB = conf_b[order_b]

    print(f"group A: {len(candA):,} (p range {pA.min():.3f}-{pA.max():.3f})")
    fam_mix_B = {f: int((y_te[candB] == f).sum()) for f in sorted(set(y_te[candB].tolist()))}
    print(f"group B: {len(candB):,}  composition={fam_mix_B}")

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    widsA = [str(wids_all[r]) for r in rows_te[candA]]
    widsB = [str(wids_all[r]) for r in rows_te[candB]]
    print(f"fetching context for {len(widsA) + len(widsB):,} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, widsA + widsB, N=a.context_words)

    rowsA = [(pA[i], 0, widsA[i], ctx.get(widsA[i], {"before": "?", "word": "?", "after": "?"}))
             for i in range(len(widsA))]
    rowsB = [(pB[i], int(y_te[candB[i]]), widsB[i],
              ctx.get(widsB[i], {"before": "?", "word": "?", "after": "?"}))
             for i in range(len(widsB))]

    render_html(rowsA, rowsB, OUT_DIR / a.out, a.top_n)


if __name__ == "__main__":
    main()
