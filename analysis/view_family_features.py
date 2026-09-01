"""Browse the content features of each family SAE (fam1 68k-jump, fam2 gradual),
independently — NOT paired (feature-pairing across SAEs is confounded by
non-identifiability). One HTML per family, features ordered so content-bearing ones
surface first, each showing its top-10 samples in context.

Feature ordering: by a 'content score' = (consistency of the modal target word) x
(is a content word, i.e. alphabetic and not a stopword). Function-word / punctuation
features sink to the bottom.

Also prints, per family, the list of confident CONTENT features (modal content-word
target in >= --consistency of the top-10) so the two families' content can be compared
directly in the console.

Usage:
  python view_family_features.py
"""
import argparse, collections, glob, html, json, os, re
import numpy as np
import pandas as pd

D = "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late"
FAMS = {
 "fam1_68kjump": f"{D}/n_only_early_and_late_olmo3_68k_fam_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_fam1_68kjump_n3000000_N=3000_k=50_lp=None",
 "fam2_gradual": f"{D}/n_only_early_and_late_olmo3_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_fam2_gradual_n3000000_N=3000_k=50_lp=None",
 "blimp": f"{D}/n_only_early_and_late_olmo3_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_blimp_n3000000_N=3000_k=50_lp=None",
 "medmcqa": f"{D}/n_only_early_and_late_olmo3_medmcqa_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_medmcqa_n3000000_N=3000_k=50_lp=None",
}
STOP = set("the a an and or of to in for on at by with is are was were be been it its this that "
           "as from not but if which who s t re ve ll m d you he she they we i his her their our "
           "your my me him them us your what when where how why all no so up out then than into".split())


def load(sae):
    df = pd.read_csv(f"{sae}/top-50_activations.csv", usecols=["feature", "act_value", "word_id"])
    df = df[df.act_value > 0]
    ctx = json.load(open(f"{sae}/top-50_words_in_context.json"))
    # LLM labels (from label_features_litellm.py), if present. Prefer the gemini file.
    labels = {}
    ldir = os.path.join(sae, "feature_labels")
    if os.path.isdir(ldir):
        prefer = os.path.join(ldir, "gemini-gemini-2.5-pro.json")
        files = [prefer] if os.path.exists(prefer) else sorted(glob.glob(f"{ldir}/*.json"))
        if files:
            try:
                labels = json.load(open(files[0]))
            except Exception:
                labels = {}
    return df, ctx, labels


def feature_rows(df, ctx, n=10):
    """Yield (fid, top_samples, modal_word, modal_frac, content_score)."""
    for fid, g in df.groupby("feature"):
        g = g.nlargest(n, "act_value")
        samples, words = [], []
        for w, av in zip(g.word_id.astype(str), g.act_value):
            e = ctx.get(w, {})
            samples.append((e.get("before", "")[-45:], e.get("word", ""), e.get("after", "")[:40], av))
            words.append((e.get("word", "") or "").strip().lower())
        wc = collections.Counter(w for w in words if w)
        modal, mc = (wc.most_common(1)[0] if wc else ("", 0))
        frac = mc / max(len(words), 1)
        is_content = bool(re.fullmatch(r"[a-z][a-z-]+", modal)) and modal not in STOP
        score = frac * (1.0 if is_content else 0.15)
        yield int(fid), samples, modal, frac, is_content, score


def print_low_consistency(fam, rows, threshold=0.5, limit=25):
    """Print features whose modal word covers < threshold of top-10, with full
    sample context, so non-lexical patterns (POS, position, punctuation, etc.)
    can be eyeballed."""
    low = [r for r in rows if r[3] < threshold]
    print(f"\n=== {fam}: {len(low)} features with modal word < {threshold:.0%} of top-10 ===")
    for fid, samples, modal, frac, is_content, score in low[:limit]:
        print(f"\n  f{fid} (modal '{modal}' only {frac:.0%})")
        for b, w, a, av in samples:
            print(f"    {av:5.1f}  ...{b}[{w}]{a}...")


def render_all(fam_rows, fam_labels):
    """One HTML, 4 side-by-side family columns. Each feature is a collapsed cell showing
    its LLM label; click to expand its top samples. The activation bucket is a per-family
    quartile, so a selected bucket shows that quartile in every column for comparison.
    Also filters by same-word% and sorts within each column."""
    esc = html.escape

    style = """<style>
body{font:13px/1.5 -apple-system,sans-serif;margin:0;padding:0;background:#fafafa}
.bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:20;
display:flex;flex-wrap:wrap;gap:16px;align-items:center;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.bar label{font-size:12px;color:#444}.bar input[type=number]{width:54px}
.bar .grp{display:flex;gap:6px;align-items:center}
.cols{display:flex;gap:8px;padding:8px;align-items:flex-start}
.col{flex:1 1 0;min-width:0;background:#f0f1f3;border-radius:6px;padding:6px;
overflow-y:auto;max-height:calc(100vh - 64px)}
.ch{position:sticky;top:0;background:#f0f1f3;font-weight:700;font-size:13px;padding:4px 4px 6px;z-index:5}
.ch small{font-weight:400;color:#888}
details.feat{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:5px 8px;margin-bottom:6px}
details.feat>summary{cursor:pointer;font-size:12px;list-style:none;outline:none}
details.feat>summary::-webkit-details-marker{display:none}
details.feat>summary::before{content:'\\25B8 ';color:#bbb}
details.feat[open]>summary::before{content:'\\25BE '}
.lab{font-weight:600}.nolab{font-weight:600;color:#aaa}.desc{color:#666;font-size:11px;margin:2px 0 4px}
.meta{color:#999;font-weight:400;font-size:10px}.pk{color:#2a7}.conf{color:#c60}
.s{font-family:ui-monospace,monospace;font-size:10.5px;white-space:pre-wrap;word-break:break-word;border-top:1px solid #f2f2f2;padding:1px 0}
.w{background:#ffe9a8;font-weight:600;padding:0 2px}.c{color:#999}.a{color:#bbb;font-size:9px}</style>"""

    bar = """<div class='bar'>
  <b>4-family feature comparison</b>
  <span class='grp'><label>same-word %:</label>
    <input type='number' id='fmin' value='0' min='0' max='100' step='5'> –
    <input type='number' id='fmax' value='100' min='0' max='100' step='5'></span>
  <span class='grp'><label>activation bucket (per-family quartile):</label>
    <select id='bucket'>
      <option value='0'>all</option>
      <option value='4'>Q4 strongest</option>
      <option value='3'>Q3</option>
      <option value='2'>Q2</option>
      <option value='1'>Q1 weakest</option>
    </select></span>
  <span class='grp'><label>sort:</label>
    <select id='sort'><option value='act'>activation ↓</option>
      <option value='frac'>same-word % ↓</option></select></span>
  <span class='grp'><label><input type='checkbox' id='expand'> expand all</label></span>
</div>"""

    parts = ["<title>Family SAE features — 4-up</title>", style, bar, "<div class='cols'>"]
    for fam, rows in fam_rows.items():
        labels = fam_labels.get(fam, {})
        feats = [(fid, samples, modal, frac, is_content,
                  max((av for *_, av in samples), default=0.0))
                 for fid, samples, modal, frac, is_content, score in rows]
        peaks = np.array([f[5] for f in feats]) if feats else np.array([0.0])
        q1, q2, q3 = (float(np.quantile(peaks, x)) for x in (0.25, 0.5, 0.75))
        def bucket(p):
            return 4 if p >= q3 else 3 if p >= q2 else 2 if p >= q1 else 1
        feats.sort(key=lambda r: -r[5])
        n_lab = sum(1 for f in feats if labels.get(str(f[0])))
        parts.append(f"<div class='col' data-fam='{esc(fam)}'>"
                     f"<div class='ch'>{esc(fam)} <small>{len(feats)} feats · "
                     f"{n_lab} labeled · Q edges {q1:.1f}/{q2:.1f}/{q3:.1f}</small></div>")
        for fid, samples, modal, frac, is_content, peak in feats:
            lab = labels.get(str(fid)) or {}
            if lab.get("error"):
                continue
                head = f"<span class='nolab'>f{fid} (label error)</span>"
                desc = ""
            elif lab.get("label") or lab.get("coherent") is not None:
                name = lab.get("label") or "(incoherent)"
                if name == "(incoherent)":
                    continue
                conf = lab.get("confidence")
                confs = f" <span class='conf'>{float(conf):.2f}</span>" if isinstance(conf, (int, float)) else ""
                head = f"<span class='lab'>{esc(str(name))}</span>{confs}"
                desc = f"<div class='desc'>{esc(str(lab.get('description','')))}</div>" if lab.get("description") else ""
            else:
                # no label yet — fall back to the modal-word heuristic
                mw = (f"{esc(modal)} ({frac:.0%})" if is_content and frac >= 0.3 else f"'{esc(modal)}' {frac:.0%}")
                head = f"<span class='nolab'>f{fid} · {mw} (unlabeled)</span>"
                desc = ""
            parts.append(
                f"<details class='feat' data-frac='{frac*100:.0f}' data-b='{bucket(peak)}' data-act='{peak:.3f}'>"
                f"<summary>{head} <span class='meta'>f{fid} · <span class='pk'>act {peak:.1f}</span> · "
                f"same-word {frac:.0%}</span></summary>{desc}")
            for b, w, a, av in samples:
                parts.append(f"<div class='s'><span class='a'>{av:.1f} </span>"
                             f"<span class='c'>{esc(b)}</span><span class='w'>{esc(w)}</span>"
                             f"<span class='c'>{esc(a)}</span></div>")
            parts.append("</details>")
        parts.append("</div>")
    parts.append("</div>")

    parts.append("""<script>
const cols=[...document.querySelectorAll('.col')];
function apply(){
  const fmin=+document.getElementById('fmin').value, fmax=+document.getElementById('fmax').value;
  const b=document.getElementById('bucket').value, sort=document.getElementById('sort').value;
  cols.forEach(col=>{
    const feats=[...col.querySelectorAll('.feat')];
    let vis=feats.filter(f=>{const fr=+f.dataset.frac;
      const ok=fr>=fmin&&fr<=fmax&&(b==='0'||f.dataset.b===b);
      f.style.display=ok?'':'none';return ok;});
    vis.sort((x,y)=> sort==='act' ? y.dataset.act-x.dataset.act : y.dataset.frac-x.dataset.frac);
    vis.forEach(f=>col.appendChild(f));
    const h=col.querySelector('.ch small'); h.dataset.base=h.dataset.base||h.textContent;
    h.textContent=h.dataset.base+' · '+vis.length+' shown';
  });
}
document.getElementById('expand').addEventListener('change',e=>{
  document.querySelectorAll('details.feat').forEach(d=>{ if(d.style.display!=='none') d.open=e.target.checked; });
});
['fmin','fmax','bucket','sort'].forEach(id=>document.getElementById(id).addEventListener('input',apply));
apply();
</script>""")

    out = "/home/nsrikant/BehaviorBoxNew/analysis/features_all_families.html"
    open(out, "w").write("\n".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consistency", type=float, default=0.4,
                    help="modal-target fraction to call a feature a confident content feature")
    ap.add_argument("--low-consistency", action="store_true",
                    help="print features with modal word < 50%% of top-10, for spotting non-lexical patterns")
    a = ap.parse_args()
    fam_rows, fam_labels = {}, {}
    for fam, sae in FAMS.items():
        df, ctx, labels = load(sae)
        rows = sorted(feature_rows(df, ctx), key=lambda r: -r[5])
        fam_rows[fam] = rows
        fam_labels[fam] = labels
        if a.low_consistency:
            print_low_consistency(fam, rows)
        n_lab = sum(1 for r in rows if labels.get(str(r[0])))
        print(f"{fam}: {len(rows)} live features, {n_lab} LLM-labeled")
    out = render_all(fam_rows, fam_labels)
    print(f"\nwrote combined viewer -> {out}")


if __name__ == "__main__":
    main()
