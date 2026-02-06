import subprocess
import json
import os
import sys
from datetime import datetime
import glob
import os

# tasks = ARC/C	HSwag	WinoG	MMLU	DROP	NQ	AGIEval	GSM8k	MMLUPro	TriviaQA
# tasks = ["arc_challenge", "blimp","hellaswag", "winogrande", "mmlu", "drop", "nq_open", "agieval", "gsm8k", "mmlu_pro", "triviaqa"]
# tasks ="arc_challenge,blimp,hellaswag,winogrande,mmlu"
tasks_type = "slow"
tasks = ["drop"]
CHECKPOINTS = {
    "olmo2_7b": {
        "name": "allenai/OLMo-2-1124-7B",
        "revisions": [
            # "stage1-step850-tokens4B",
            # "stage1-step9000-tokens38B",
            # "stage1-step47000-tokens198B",
            # "stage1-step94000-tokens395B",
            # "stage1-step235000-tokens986B",
            # "stage1-step470000-tokens1972B",
            # "stage1-step705000-tokens2957B",
            # "stage1-step847000-tokens3553B",
            # "stage2-ingredient1-step1000-tokens5B",
            # "stage2-ingredient1-step7000-tokens30B",
            "main"
        ],
    }
}



import pandas as pd

def parse_file(filepath):
    with open(filepath, "r") as fp:
        lines = fp.readlines()



    idx = lines.index("|                           Tasks                            |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|\n")
    idx2 = lines.index("|      Groups      |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|\n")
    idx3 = lines.index("==================================================\n")
    detailed_cols = list(filter(None, list(map(str.strip, lines[idx].split("|")))))
    cols = list(filter(None, list(map(str.strip, lines[idx2].split("|")))))
    detailed_df = pd.DataFrame(columns=detailed_cols)
    df = pd.DataFrame(columns=cols)
    for line in lines[idx+2:idx2]:
        if line.strip():
            
            eles = list(map(str.strip, line.strip().split("|")))[1:-1]
            
            new_eles = [ele for ele in eles if ele not in ['↑', '±']]
            try:
                detailed_df.loc[len(detailed_df)] = new_eles
            except:
                print(line)



    for line in lines[idx2+2:-6]:
        
        if line.strip():
            eles = list(map(str.strip, line.strip().split("|")))[1:-1]
            new_eles = [ele for ele in eles if ele not in ['↑', '±']]
            try:
                df.loc[len(df)] = new_eles
            except:
                print(line)

    return detailed_df, df

def run_evals():
    for model_key, model_info in CHECKPOINTS.items():
        name = model_info["name"]
        all_results = {}

        os.makedirs("logs", exist_ok=True)
        os.makedirs(f"logs/{model_key}/{tasks_type}", exist_ok=True)
        
        # Create logs folder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{model_key}/{tasks_type}/{name}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        total = len(model_info["revisions"])
        
        for i, revision in enumerate(model_info["revisions"], 1):
            
            cmd = [
                "lm_eval",
                "--model", "vllm",
                "--model_args", f"pretrained={name},revision={revision},gpu_memory_utilization=0.8,enable_chunked_prefill=False",
                "--tasks", ",".join(tasks),
                "--batch_size", "auto",
                "--gen_kwargs", "min_tokens=1",
                "--log_samples",
                "--output_path", f"drop_test"
            ]
            
            print(f"\n[{i}/{total}] Running: {revision}")
            print(f"Command: {' '.join(cmd)}")
            
            start_time = datetime.now()
            log_file = os.path.join(log_dir, f"{revision}.log")
            output_lines = []
            
            # Stream output in realtime
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Start: {start_time}\n")
                f.write("=" * 50 + "\n")
                f.flush()
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                
                for line in process.stdout:
                    # Print to console
                    print(line, end="")
                    sys.stdout.flush()
                    # Write to log file
                    f.write(line)
                    f.flush()
                    # Collect for parsing
                    output_lines.append(line)
                
                process.wait()
                
                end_time = datetime.now()
                duration = end_time - start_time
                
                f.write("=" * 50 + "\n")
                f.write(f"End: {end_time}\n")
                f.write(f"Duration: {duration}\n")
                f.write(f"Return code: {process.returncode}\n")
            
            

        folder = f"logs/{model_key}"
        results_folder = f"/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results/{model_key}/{tasks_type}"
        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(f"{results_folder}/grouped", exist_ok=True)
        for filename in glob.glob(f"{folder}**/*.log"):
            outputfn = filename.split("/")[-1][:-4]
            detailed_df, df = parse_file(filename)
            with open(f"{results_folder}/{outputfn}.csv", "w") as f:
                detailed_df.to_csv(f, index=False)
            with open(f"{results_folder}/grouped/{outputfn}.csv", "w") as f:
                df.to_csv(f, index=False)


if __name__ == "__main__":
    run_evals()