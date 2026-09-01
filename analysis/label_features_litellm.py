"""Batch-label SAE features with an LLM via litellm.

For each live feature, the model is shown the tokens where it activates most strongly
(activating token wrapped in «...», with the activation value) and asked whether they
form a coherent pattern — lexical (same word/inflection), morphological, syntactic
(POS / grammatical role), topical (shared context domain), format (markup/punctuation),
or incoherent. Returns a concise label + description + category + confidence as JSON.

Resumable: writes to <sae_dir>/feature_labels/<model>.json and skips features already
present unless --replace. Concurrent via a thread pool; retries with backoff.

Usage:
  python label_features_litellm.py --sae-dir <dir> --model gemini/gemini-2.5-pro
  python label_features_litellm.py --sae-dir <dir> --limit 20        # test run
Requires the provider key in the environment, e.g. GEMINI_API_KEY for gemini/*.
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import litellm


def load_dotenv(path=None):
    """Minimal .env loader: KEY=VALUE lines into os.environ (does not overwrite existing)."""
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

SYSTEM_PROMPT = """You are an interpretability researcher labeling features from a sparse autoencoder (SAE) trained on contextual token embeddings of pretraining text. Each feature activates on specific tokens in context. Below are the tokens where ONE feature activates most strongly; in each line the ACTIVATING token is wrapped in «double angle brackets» and prefixed by its activation strength.

Decide whether these activations form a COHERENT pattern. Coherence can come from EITHER:
 (a) the ACTIVATING TOKENS themselves — the same word or its inflections (lexical), a morphological pattern, or a part-of-speech / grammatical role (syntactic) / any other pattern; OR
 (b) the SURROUNDING CONTEXTS — a shared topic, domain, or register (topical), or a formatting / markup / punctuation role (format) — even when the activating tokens are generic stop-words or punctuation.

A feature is coherent if EITHER holds. Do NOT reject a feature just because the activating tokens look generic — first check whether the contexts share a topic or register. A feature is INCOHERENT only when neither the tokens nor the contexts share any theme.

Respond with ONLY a JSON object, no other text:
{
  "coherent": true or false,
  "label": "<= 8 words naming the feature; the single word if lexical, otherwise the shared pattern>",
  "confidence": a number from 0.0 to 1.0
}
If incoherent: set coherent=false, label="", and a low confidence."""


def clean(s):
    return (s or "").replace("Ġ", " ").replace("Ċ", " ").replace("\t", " ").replace("\n", " ").strip()


def build_user_prompt(rows, before_chars=90, after_chars=60):
    lines = []
    for av, before, word, after in rows:
        b = clean(before)[-before_chars:]
        w = clean(word)
        a = clean(after)[:after_chars]
        lines.append(f"- (act {av:.1f}) {b} «{w}» {a}")
    return ("Feature activations (strongest first):\n" + "\n".join(lines)
            + "\n\nReturn the JSON object.")


def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def label_one(fid, rows, model, call_kwargs=None, retries=3):
    user = build_user_prompt(rows)
    call_kwargs = call_kwargs or {}
    for attempt in range(retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.0,
                # gemini-2.5-pro is a reasoning model: it spends tokens on internal
                # thinking before the answer, so the budget must cover reasoning + JSON
                # or content comes back None with finish_reason=length.
                max_tokens=2048,
                **call_kwargs,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError(f"empty content (finish={resp.choices[0].finish_reason})")
            parsed = parse_json(content)
            if parsed is not None:
                parsed["n_samples"] = len(rows)
                return fid, parsed
        except Exception as e:
            if attempt == retries - 1:
                return fid, {"error": str(e)[:200]}
            time.sleep(2 ** attempt + 0.5)
    return fid, {"error": "unparseable_response"}


def feature_samples(sae_dir, n_samples, min_acts):
    """Return {fid: (rows, modal_token, same_frac)} for features with >= min_acts samples.
    rows = [(act, before, word, after), ...]."""
    import collections
    acts = pd.read_csv(f"{sae_dir}/top-50_activations.csv",
                       usecols=["feature", "act_value", "word_id"])
    acts = acts[acts.act_value > 0]
    wic = json.load(open(f"{sae_dir}/top-50_words_in_context.json"))
    out = {}
    for fid, g in acts.groupby("feature"):
        g = g.nlargest(n_samples, "act_value")
        if len(g) < min_acts:
            continue
        rows, words = [], []
        for wid, av in zip(g.word_id.astype(str), g.act_value):
            e = wic.get(wid, {})
            rows.append((float(av), e.get("before", ""), e.get("word", ""), e.get("after", "")))
            words.append((e.get("word", "") or "").strip())
        wc = collections.Counter(w for w in words if w)
        modal, mc = (wc.most_common(1)[0] if wc else ("", 0))
        out[int(fid)] = (rows, modal, mc / max(len(words), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--model", default="gemini/gemini-2.5-pro")
    ap.add_argument("--n-samples", type=int, default=10, help="top samples shown to the LLM")
    ap.add_argument("--min-acts", type=int, default=10, help="skip features with fewer samples")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-same-frac", type=float, default=0.5,
                    help="features with a single token >= this fraction of samples are "
                         "auto-labeled with that token (no LLM call); only features below "
                         "it are sent to the model.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all features (LLM ones)")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--env", default=None, help="path to .env with the API key")
    a = ap.parse_args()
    load_dotenv(a.env)

    # Route through the LiteLLM proxy if configured (LITELLM_BASE_URL/API_KEY in .env),
    # otherwise call the provider directly (needs e.g. GEMINI_API_KEY).
    model = a.model
    call_kwargs = {}
    base_url = os.environ.get("LITELLM_BASE_URL")
    if base_url:
        if not model.startswith("litellm_proxy/"):
            model = "litellm_proxy/" + model
        call_kwargs = {"api_key": os.environ.get("LITELLM_API_KEY"), "base_url": base_url}
        print(f"using LiteLLM proxy at {base_url}  (model={model})", flush=True)

    out_dir = os.path.join(a.sae_dir, "feature_labels")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, a.model.replace("/", "-") + ".json")
    labels = json.load(open(out_file)) if (os.path.exists(out_file) and not a.replace) else {}

    samples = feature_samples(a.sae_dir, a.n_samples, a.min_acts)
    auto, todo = [], []
    for fid, (rows, modal, same_frac) in samples.items():
        if str(fid) in labels:
            continue
        if same_frac >= a.max_same_frac:
            auto.append((fid, modal, same_frac, len(rows)))
        else:
            todo.append((fid, rows))
    # auto-label the single-token features with the token itself, no API call
    for fid, modal, same_frac, n in auto:
        labels[str(fid)] = {"coherent": True, "label": modal, "confidence": 1.0,
                            "auto": True, "same_frac": round(same_frac, 2), "n_samples": n}
    if a.limit:
        todo = todo[:a.limit]
    print(f"{os.path.basename(a.sae_dir)}: {len(samples)} eligible · "
          f"{len(auto)} auto-labeled (same token >= {a.max_same_frac:.0%}) · "
          f"{len(todo)} to LLM-label  (model={a.model})", flush=True)

    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(label_one, f, r, model, call_kwargs) for f, r in todo]
        for fut in as_completed(futs):
            fid, res = fut.result()
            with lock:
                labels[str(fid)] = res
                done += 1
                if done % 25 == 0 or done == len(todo):
                    json.dump(labels, open(out_file, "w"), indent=1)
                    print(f"  {done}/{len(todo)} LLM-labeled", flush=True)
    json.dump(labels, open(out_file, "w"), indent=1)
    n_err = sum(1 for v in labels.values() if "error" in v)
    n_coh = sum(1 for v in labels.values() if v.get("coherent"))
    print(f"DONE -> {out_file}  ({len(labels)} total, {n_coh} coherent, {n_err} errors)")


if __name__ == "__main__":
    main()
