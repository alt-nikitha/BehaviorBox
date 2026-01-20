import pandas as pd
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

checkpoints = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    10000,
    70000,
    100000,
    140000

]
models = ["160m", "6.9b"]
trends = ["increasing", "increase-decrease", "increase-stagnate"]
for model in models:
    for trend in trends:
        INPUT_CSV = f"/home/nsrikant/BehaviorBoxNew/analysis/ad_hoc/trend_outputs/Pythia{model}/{trend}/acquired_topic_summary.csv"

        OUTPUT_HTML = INPUT_CSV.replace(".csv", "_feature_acquisition_diagram.html")

        EARLY_MAX = 512                    # <= 512
        MID_MIN = 1000                     # >= 1000 and < 70000
        MID_MAX = 70000
        LATE_MIN = 70000                   # >= 70000

        # =====================================================
        # LOAD CSV
        # =====================================================
        df = pd.read_csv(INPUT_CSV)

        # Standardize column names

        if "Topic" not in df.columns:
            raise ValueError("CSV must contain a 'Topic' column")
        if "First Acquired Checkpoint" not in df.columns:
            raise ValueError("CSV must contain an 'acquired_checkpoint' column")

        # Ensure numeric
        df["First Acquired Checkpoint"] = pd.to_numeric(df["First Acquired Checkpoint"], errors="coerce")


        # =====================================================
        # CLASSIFY FEATURES
        # =====================================================
        early = []
        mid = []
        late = []

        for _, row in df.iterrows():
            name = row["Topic"]
            ckpt = row["First Acquired Checkpoint"]

            if checkpoints[ckpt] <= EARLY_MAX:
                early.append((name, ckpt))
            elif MID_MIN <= checkpoints[ckpt] < MID_MAX:
                mid.append((name, ckpt))
            else:  # ckpt >= LATE_MIN
                late.append((name, ckpt))

        # Sort by checkpoint
        early.sort(key=lambda x: x[1])
        mid.sort(key=lambda x: x[1])
        late.sort(key=lambda x: x[1])


        # =====================================================
        # HTML TEMPLATE
        # =====================================================
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8" />
        <title>Feature Acquisition Diagram</title>

        <style>
        body {{
            font-family: Arial, sans-serif;
            background: #fafafa;
            padding: 30px;
        }}
        h1 {{
            text-align: center;
        }}
        .lane-container {{
            display: flex;
            justify-content: space-around;
            margin-top: 40px;
        }}
        .lane {{
            width: 28%;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}
        .lane h2 {{
            text-align: center;
        }}
        .item {{
            background: #eef2ff;
            margin: 8px 0;
            padding: 8px;
            border-radius: 8px;
            font-size: 14px;
        }}
        </style>

        </head>
        <body>

        <h1>Feature Acquisition Diagram</h1>
        <p style="text-align:center;">Automatic Early / Mid / Late Breakdown</p>

        <div class="lane-container">

        <!-- EARLY -->
        <div class="lane">
            <h2>Early (≤ 512)</h2>
            {"".join([f'<div class="item">{name}</div>' for name, ckpt in early])}
        </div>

        <!-- MID -->
        <div class="lane">
            <h2>Mid (1000–70000)</h2>
            {"".join([f'<div class="item">{name}</div>' for name, ckpt in mid])}
        </div>

        <!-- LATE -->
        <div class="lane">
            <h2>Late (≥ 70000)</h2>
            {"".join([f'<div class="item">{name}</div>' for name, ckpt in late])}
        </div>

        </div>

        </body>
        </html>
        """

        # =====================================================
        # WRITE OUTPUT
        # =====================================================
        Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
        print(f"HTML diagram written to {OUTPUT_HTML}")
