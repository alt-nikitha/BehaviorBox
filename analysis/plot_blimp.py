import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 5)

# ============================================================================
# LOAD BLIMP SCORES FROM RESULTS
# ============================================================================
os.makedirs('blimp_plots', exist_ok=True)

def get_model_checkpoints(model_name):
    """Get all checkpoints for a model, sorted in order"""
    base_path = f"/home/nsrikant/BehaviorBoxNew/results/blimp_full/"
    checkpoints = []
    
    for folder in sorted(os.listdir(base_path)):
        if folder.startswith(model_name):
            checkpoints.append(folder)
    
    # Sort by step number for proper ordering
    def extract_step(checkpoint_name):
        """Extract step number for sorting"""
        if 'step' in checkpoint_name:
            try:
                step_str = checkpoint_name.split('step')[-1]
                return int(step_str)
            except:
                return float('inf')
        return float('inf')
    
    # Separate final model from step models
    final_models = [cp for cp in checkpoints if extract_step(cp) == float('inf')]
    step_models = [cp for cp in checkpoints if extract_step(cp) != float('inf')]
    
    # Sort step models by step number
    step_models_sorted = sorted(step_models, key=extract_step)
    
    # Combine: steps first (in order), then final model last
    return step_models_sorted + final_models

def load_blimp_scores(model_name, checkpoint_folder):
    """Load BLIMP scores from checkpoint folder"""
    json_path = f"/home/nsrikant/BehaviorBoxNew/results/blimp_full/{checkpoint_folder}/blimp_full_summary_seq_logprob.json"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: Could not find {json_path}")
        return None

# ============================================================================
# EXTRACT PERFORMANCE DATA
# ============================================================================

print("Loading Pythia-160m checkpoints...")
checkpoints_160m = get_model_checkpoints("pythia-160m")
print(f"Found {len(checkpoints_160m)} checkpoints:")
for cp in checkpoints_160m:
    print(f"  - {cp}")

print("\nLoading Pythia-6.9b checkpoints...")
checkpoints_6_9b = get_model_checkpoints("pythia-6_9b")
print(f"Found {len(checkpoints_6_9b)} checkpoints:")
for cp in checkpoints_6_9b:
    print(f"  - {cp}")

# Load scores
def load_model_scores(checkpoints):
    """Load all scores for a model"""
    overall_scores = []
    subset_scores = {}
    
    for checkpoint in checkpoints:
        data = load_blimp_scores("", checkpoint)
        
        if data is None:
            continue
        
        # Get overall accuracy
        overall_acc = data.get('overall_accuracy', 0)
        overall_scores.append(overall_acc)
        
        # Get subset accuracies - FIXED: initialize once, append for each checkpoint
        for subset_name, score in data["per_split_accuracy"].items():
            if subset_name not in subset_scores:
                subset_scores[subset_name] = []
            subset_scores[subset_name].append(score)

    return overall_scores, subset_scores

print("\nLoading scores...")
overall_160m, subsets_160m = load_model_scores(checkpoints_160m)
overall_6_9b, subsets_6_9b = load_model_scores(checkpoints_6_9b)

print(f"Pythia-160m: {len(overall_160m)} checkpoints loaded")
print(f"Pythia-6.9b: {len(overall_6_9b)} checkpoints loaded")

# Extract step numbers from checkpoint names for x-axis labels
def extract_step_number(checkpoint_name):
    """Extract step number from checkpoint folder name"""
    # Format is typically "pythia-160m-step1000" or similar
    if 'step' in checkpoint_name:
        step_str = checkpoint_name.split('step')[-1]
        return f"step {step_str}"
    return checkpoint_name

x_labels_160m = [extract_step_number(cp) for cp in checkpoints_160m[:len(overall_160m)]]
x_labels_6_9b = [extract_step_number(cp) for cp in checkpoints_6_9b[:len(overall_6_9b)]]

# ============================================================================
# PLOT 1: OVERALL PERFORMANCE COMPARISON
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(range(len(overall_160m)), overall_160m, 
        marker='o', linewidth=2.5, markersize=8, 
        label='Pythia-160m', color='#1f77b4', alpha=0.8)

ax.plot(range(len(overall_6_9b)), overall_6_9b, 
        marker='s', linewidth=2.5, markersize=8,
        label='Pythia-6.9b', color='#ff7f0e', alpha=0.8)

ax.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
ax.set_ylabel('BLIMP Performance (Accuracy)', fontsize=12, fontweight='bold')
ax.set_title('Overall BLIMP Performance Across Training Checkpoints', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis ticks and labels (show every Nth label to avoid crowding)
max_checkpoints = max(len(overall_160m), len(overall_6_9b))
step_size = max(1, max_checkpoints // 10)  # Show ~10 labels

x_ticks_160m = range(0, len(overall_160m), step_size)
x_labels_display_160m = [x_labels_160m[i] for i in x_ticks_160m if i < len(x_labels_160m)]

ax.set_xticks(x_ticks_160m)
ax.set_xticklabels(x_labels_display_160m, rotation=45, ha='right')

ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('blimp_plots/blimp_overall_performance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: blimp_overall_performance.png")
plt.close()

# ============================================================================
# PLOT 2: SUBSET PERFORMANCE COMPARISON (SUBPLOTS) - PART 1
# ============================================================================

# Get all subset names
all_subsets = sorted(set(list(subsets_160m.keys()) + list(subsets_6_9b.keys())))

print(f"\nFound {len(all_subsets)} BLIMP subsets:")
for subset in all_subsets:
    print(f"  - {subset}")

# Split subsets into two groups
mid_point = (len(all_subsets) + 1) // 2
subsets_part1 = all_subsets[:mid_point]
subsets_part2 = all_subsets[mid_point:]

# PART 1: First half of subsets
n_subsets_1 = len(subsets_part1)
n_cols = 4
n_rows_1 = (n_subsets_1 + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows_1, n_cols, figsize=(16, n_rows_1 * 3.5))
axes = axes.flatten()

for idx, subset_name in enumerate(subsets_part1):
    ax = axes[idx]
    
    # Get performance data
    perf_160m_subset = subsets_160m.get(subset_name, [])
    perf_6_9b_subset = subsets_6_9b.get(subset_name, [])
    
    # Plot
    if perf_160m_subset:
        ax.plot(range(len(perf_160m_subset)), perf_160m_subset,
                marker='o', linewidth=2, markersize=6,
                label='Pythia-160m', color='#1f77b4', alpha=0.8)
    
    if perf_6_9b_subset:
        ax.plot(range(len(perf_6_9b_subset)), perf_6_9b_subset,
                marker='s', linewidth=2, markersize=6,
                label='Pythia-6.9b', color='#ff7f0e', alpha=0.8)
    
    ax.set_title(subset_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Checkpoint', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels with step names (show every Nth label to avoid crowding)
    max_subset_checkpoints = max(len(perf_160m_subset), len(perf_6_9b_subset))
    step_size_subset = max(1, max_subset_checkpoints // 5)  # Show ~5 labels per subplot
    
    if perf_160m_subset:
        x_ticks_subset = range(0, len(perf_160m_subset), step_size_subset)
        x_labels_subset = [x_labels_160m[i] for i in x_ticks_subset if i < len(x_labels_160m)]
        ax.set_xticks(x_ticks_subset)
        ax.set_xticklabels(x_labels_subset, rotation=45, ha='right', fontsize=8)
    elif perf_6_9b_subset:
        x_ticks_subset = range(0, len(perf_6_9b_subset), step_size_subset)
        x_labels_subset = [x_labels_6_9b[i] for i in x_ticks_subset if i < len(x_labels_6_9b)]
        ax.set_xticks(x_ticks_subset)
        ax.set_xticklabels(x_labels_subset, rotation=45, ha='right', fontsize=8)
    
    if idx == 0:
        ax.legend(fontsize=10, loc='lower right')

# Remove extra subplots
for idx in range(n_subsets_1, len(axes)):
    fig.delaxes(axes[idx])

fig.suptitle('BLIMP Subset Performance Across Training Checkpoints (Part 1)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('blimp_plots/blimp_subset_performance_part1.png', dpi=300, bbox_inches='tight')
print("✅ Saved: blimp_subset_performance_part1.png")
plt.close()

# ============================================================================
# PLOT 2B: SUBSET PERFORMANCE COMPARISON (SUBPLOTS) - PART 2
# ============================================================================

# PART 2: Second half of subsets
n_subsets_2 = len(subsets_part2)
n_rows_2 = (n_subsets_2 + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows_2, n_cols, figsize=(16, n_rows_2 * 3.5))
axes = axes.flatten()

for idx, subset_name in enumerate(subsets_part2):
    ax = axes[idx]
    
    # Get performance data
    perf_160m_subset = subsets_160m.get(subset_name, [])
    perf_6_9b_subset = subsets_6_9b.get(subset_name, [])
    
    # Plot
    if perf_160m_subset:
        ax.plot(range(len(perf_160m_subset)), perf_160m_subset,
                marker='o', linewidth=2, markersize=6,
                label='Pythia-160m', color='#1f77b4', alpha=0.8)
    
    if perf_6_9b_subset:
        ax.plot(range(len(perf_6_9b_subset)), perf_6_9b_subset,
                marker='s', linewidth=2, markersize=6,
                label='Pythia-6.9b', color='#ff7f0e', alpha=0.8)
    
    ax.set_title(subset_name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Checkpoint', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels with step names (show every Nth label to avoid crowding)
    max_subset_checkpoints = max(len(perf_160m_subset), len(perf_6_9b_subset))
    step_size_subset = max(1, max_subset_checkpoints // 5)  # Show ~5 labels per subplot
    
    if perf_160m_subset:
        x_ticks_subset = range(0, len(perf_160m_subset), step_size_subset)
        x_labels_subset = [x_labels_160m[i] for i in x_ticks_subset if i < len(x_labels_160m)]
        ax.set_xticks(x_ticks_subset)
        ax.set_xticklabels(x_labels_subset, rotation=45, ha='right', fontsize=8)
    elif perf_6_9b_subset:
        x_ticks_subset = range(0, len(perf_6_9b_subset), step_size_subset)
        x_labels_subset = [x_labels_6_9b[i] for i in x_ticks_subset if i < len(x_labels_6_9b)]
        ax.set_xticks(x_ticks_subset)
        ax.set_xticklabels(x_labels_subset, rotation=45, ha='right', fontsize=8)
    
    if idx == 0:
        ax.legend(fontsize=10, loc='lower right')

# Remove extra subplots
for idx in range(n_subsets_2, len(axes)):
    fig.delaxes(axes[idx])

fig.suptitle('BLIMP Subset Performance Across Training Checkpoints (Part 2)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('blimp_plots/blimp_subset_performance_part2.png', dpi=300, bbox_inches='tight')
print("✅ Saved: blimp_subset_performance_part2.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nPythia-160m:")
print(f"  Initial BLIMP accuracy: {overall_160m[0]:.4f}")
print(f"  Final BLIMP accuracy:   {overall_160m[-1]:.4f}")
print(f"  Improvement:            {(overall_160m[-1] - overall_160m[0]):.4f}")

print("\nPythia-6.9b:")
print(f"  Initial BLIMP accuracy: {overall_6_9b[0]:.4f}")
print(f"  Final BLIMP accuracy:   {overall_6_9b[-1]:.4f}")
print(f"  Improvement:            {(overall_6_9b[-1] - overall_6_9b[0]):.4f}")

print("\nSubset Statistics (Final Checkpoint):")
print("-" * 80)

for subset_name in all_subsets:
    perf_160m_subset = subsets_160m.get(subset_name, [])
    perf_6_9b_subset = subsets_6_9b.get(subset_name, [])
    
    acc_160m = perf_160m_subset[-1] if perf_160m_subset else 0
    acc_6_9b = perf_6_9b_subset[-1] if perf_6_9b_subset else 0
    
    print(f"{subset_name:35s} | 160m: {acc_160m:.4f} | 6.9b: {acc_6_9b:.4f}")

print("\n✅ All plots saved to blimp_plots/")
print(f"   - blimp_overall_performance.png")
print(f"   - blimp_subset_performance_part1.png (subsets 1-{len(subsets_part1)})")
print(f"   - blimp_subset_performance_part2.png (subsets {len(subsets_part1)+1}-{len(all_subsets)})")