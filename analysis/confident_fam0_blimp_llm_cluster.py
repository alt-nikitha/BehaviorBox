"""1. Take tokens confidently + correctly classified as fam0 (late-reasoning) and, separately,
   as fam2 (blimp), by the saved logreg (no retraining).
2. Cluster each group's FULL 768-d Longformer embeddings (not the 2D d1,d2 projection) via
   KMeans, K chosen per group via the elbow method.
3. Label each cluster with an LLM (litellm, same convention as label_features_litellm.py --
   gemini/gemini-2.5-pro routed through the project's LiteLLM proxy), showing it RICHER context
   than the 10-word window used everywhere else this session (default 40 words before/after),
   and asking it to name the shared theme/topic/register or say the cluster is incoherent.

Output: one HTML with two sections (FAM0 confident clusters | BLIMP confident clusters),
each cluster showing its LLM label + confidence + up to --max-show member contexts with the
target token highlighted.

Loads the saved clf+scaler, reproduces the same exclusive sample/split.

Usage:
  python confident_fam0_blimp_llm_cluster.py --model fam0_logreg_weights.pkl --n-confident 2000
"""
import argparse, html, json, os, pickle, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)
from logreg_dim_cluster_per_family import find_knee

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DIR = Path(__file__).resolve().parent
SIDE_COLOR = {0: "#2a78d6", 2: "#1baf7a"}


def load_dotenv(path=None):
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        for cand in (os.path.join(root, "scripts", "env_configs", ".env"),
                     os.path.join(here, ".env"), os.path.join(root, ".env")):
            if os.path.exists(cand):
                path = cand
                break
    if not path or not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def clean(s):
    return (s or "").replace("Ġ", " ").replace("Ċ", " ").replace("\t", " ").replace("\n", " ").strip()


SYSTEM_PROMPT = """You are an interpretability researcher analyzing a cluster of pretraining-text
token examples produced by k-means on their contextual embeddings. In each example, the
clustered token is wrapped in «double angle brackets». You are given generous surrounding
context (dozens of words on each side), not just the target token.

Decide whether these examples share a COHERENT pattern -- this can be a TOPIC/domain, a
register/genre/style (e.g. legal text, source code, narrative fiction, academic writing), a
formatting/document-type pattern, or a lexical/syntactic property of the target tokens
themselves. Use the full context, not just the target token, to judge topic/register.

Respond with ONLY a JSON object, no other text:
{
  "coherent": true or false,
  "label": "<= 8 words naming the shared pattern",
  "description": "<= 30 words explaining what ties the examples together, or why they don't",
  "confidence": a number from 0.0 to 1.0
}"""


def build_user_prompt(rows):
    lines = []
    for i, (before, word, after) in enumerate(rows, 1):
        lines.append(f"{i}. ...{clean(before)} «{clean(word)}» {clean(after)}...")
    return "Cluster examples:\n" + "\n".join(lines) + "\n\nReturn the JSON object."


def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def llm_label(rows, model, call_kwargs, retries=3):
    user = build_user_prompt(rows)
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=2048, **call_kwargs,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError(f"empty content (finish={resp.choices[0].finish_reason})")
            parsed = parse_json(content)
            if parsed is not None:
                return parsed
        except Exception as e:
            if attempt == retries - 1:
                return {"error": str(e)[:300]}
            time.sleep(2 ** attempt + 0.5)
    return {"error": "unparseable_response"}


def render_html(side_clusters, out_path, max_show):
    esc = html.escape
    parts = [f"""<title>LLM-labeled clusters: confident fam0 vs confident blimp</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px}}
.meta-top{{color:#666;font-size:13px;margin-bottom:16px}}
.cols{{display:flex;gap:14px;overflow-x:auto;padding-bottom:8px}}
.col{{flex:1;min-width:280px;max-width:380px}}
.col h3{{font-size:13px;margin:0 0 8px;position:sticky;top:0;background:#fafafa;padding:6px 0;
color:#333}}
.tok{{background:#fff;border:1px solid #e3e3e3;border-radius:6px;margin-bottom:6px;padding:6px 8px}}
.ctx{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;white-space:pre-wrap;word-break:break-word}}
.before,.after{{color:#777}}
.w{{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}}
.llmlabel{{font-weight:600;color:#0a5;font-size:12.5px}}
.llmdesc{{font-weight:400;color:#555;font-size:11.5px;display:block;margin-top:2px}}
.incoherent .llmlabel{{color:#b33}}
</style>
<h1>LLM-labeled clusters: confidently-correct fam0 vs confidently-correct blimp</h1>
<div class="meta-top">Clustered on FULL 768-d Longformer embeddings (K via elbow per side).
Each cluster labeled by gemini-2.5-pro given generous context per example (not just 10 words).
</div>
"""]
    for c, clusters in side_clusters.items():
        parts.append(f'<h2 style="color:{SIDE_COLOR[c]}">fam{c} ({esc(FAM_LABELS.get(c, str(c)))}) '
                     f'-- confidently correct</h2>')
        parts.append('<div class="cols">')
        for rank, cl in enumerate(clusters, 1):
            lbl = cl["llm"]
            coherent = lbl.get("coherent", False)
            cls = "" if coherent else "incoherent"
            label_text = lbl.get("label") or lbl.get("error") or "(no label)"
            desc = lbl.get("description", "")
            conf = lbl.get("confidence")
            conf_str = f" (conf {conf:.2f})" if isinstance(conf, (int, float)) else ""
            shown = cl["examples"][:max_show]
            parts.append(f'<div class="col {cls}"><h3>cluster {rank}/{len(clusters)} '
                         f'n={cl["n"]:,} (showing {len(shown)})<br>'
                         f'<span class="llmlabel">{esc(label_text)}{conf_str}</span>'
                         f'<span class="llmdesc">{esc(desc)}</span></h3>')
            for e in shown:
                parts.append(f'<div class="tok"><div class="ctx">'
                             f'<span class="before">{esc(e["before"])}</span>'
                             f'<span class="w">{esc(e["word"])}</span>'
                             f'<span class="after">{esc(e["after"])}</span></div></div>')
            parts.append('</div>')
        parts.append('</div>')
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--llm-model", default="gemini/gemini-2.5-pro")
    ap.add_argument("--n-confident", type=int, default=2000,
                     help="confidently+correctly classified tokens kept per class")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-llm-examples", type=int, default=15,
                     help="nearest-centroid examples shown to the LLM per cluster")
    ap.add_argument("--max-show", type=int, default=20, help="examples shown in the HTML")
    ap.add_argument("--llm-context-words", type=int, default=40,
                     help="words before/after fetched for the LLM prompt (vs 10 used elsewhere)")
    ap.add_argument("--display-context-words", type=int, default=20,
                     help="words before/after shown in the HTML")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="confident_fam0_blimp_llm_cluster.html")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env", default=None)
    a = ap.parse_args()
    load_dotenv(a.env)

    llm_model = a.llm_model
    call_kwargs = {}
    base_url = os.environ.get("LITELLM_BASE_URL")
    if base_url:
        if not llm_model.startswith("litellm_proxy/"):
            llm_model = "litellm_proxy/" + llm_model
        call_kwargs = {"api_key": os.environ.get("LITELLM_API_KEY"), "base_url": base_url}
        print(f"using LiteLLM proxy at {base_url}  (model={llm_model})", flush=True)

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}")

    fam_names, F, members = build_families(0.45)
    R, D_, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D_, valid, sargs["min_r"], sargs["max_l1"],
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
    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))

    side_clusters = {}
    for target_class in [0, 2]:
        col = classes.index(target_class)
        m = (y_te == target_class) & (pred == target_class)
        cand = np.where(m)[0]
        cand = cand[np.argsort(-proba[cand, col])[: a.n_confident]]
        print(f"\nfam{target_class}: {m.sum():,} confidently+correctly classified; "
              f"using top {len(cand):,} by p (range {proba[cand, col].min():.3f}-"
              f"{proba[cand, col].max():.3f})")

        X_side = X_te[cand]
        ks_range = list(range(a.k_min, a.k_max + 1))
        inertias = []
        for k in ks_range:
            km = KMeans(n_clusters=k, random_state=a.seed, n_init=5).fit(X_side)
            inertias.append(km.inertia_)
        knee = find_knee(ks_range, inertias)
        print(f"  inertias={[f'{v:.0f}' for v in inertias]}  chosen K={knee}")

        km = KMeans(n_clusters=knee, random_state=a.seed, n_init=10).fit(X_side)
        lbl = km.labels_
        dist = np.linalg.norm(X_side - km.cluster_centers_[lbl], axis=1)
        sizes = np.bincount(lbl, minlength=knee)
        order = np.argsort(-sizes)

        word_ids_side = [str(wids_all[rows_te[i]]) for i in cand]
        print(f"  fetching LLM-context ({a.llm_context_words}w) + display-context "
              f"({a.display_context_words}w) for {len(word_ids_side):,} tokens...")
        ctx_llm = get_words_in_context(INPUT_FEATURES, word_ids_side, N=a.llm_context_words)
        ctx_disp = get_words_in_context(INPUT_FEATURES, word_ids_side, N=a.display_context_words)

        cluster_jobs = []
        cluster_meta = []
        for cl in order:
            mm = np.where(lbl == cl)[0]
            m_sorted = mm[np.argsort(dist[mm])]
            wids_sorted = [word_ids_side[i] for i in m_sorted]
            llm_rows = [(ctx_llm.get(w, {"before": "", "word": "", "after": ""})["before"],
                        ctx_llm.get(w, {"before": "", "word": "", "after": ""})["word"],
                        ctx_llm.get(w, {"before": "", "word": "", "after": ""})["after"])
                       for w in wids_sorted[: a.n_llm_examples]]
            disp_examples = [ctx_disp.get(w, {"before": "?", "word": "?", "after": "?"})
                             for w in wids_sorted]
            cluster_jobs.append(llm_rows)
            cluster_meta.append({"n": len(mm), "examples": disp_examples})

        print(f"  LLM-labeling {len(cluster_jobs)} clusters (model={a.llm_model})...")
        results = [None] * len(cluster_jobs)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(llm_label, rows_, llm_model, call_kwargs): i
                    for i, rows_ in enumerate(cluster_jobs)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                print(f"    cluster {i+1}: {results[i]}", flush=True)

        clusters = []
        for meta, res in zip(cluster_meta, results):
            clusters.append({**meta, "llm": res or {"error": "no result"}})
        side_clusters[target_class] = clusters

    render_html(side_clusters, OUT_DIR / a.out, a.max_show)


if __name__ == "__main__":
    main()
