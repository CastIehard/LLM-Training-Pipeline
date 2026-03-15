import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(results_dir):
    results = []
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory {results_dir} does not exist.")
        return results

    for run_dir in results_path.iterdir():
        if run_dir.is_dir():
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    # Add directory name as fallback model name if missing
                    if 'answer_model' in data and 'model' in data['answer_model']:
                        model_name = data['answer_model']['model']
                    else:
                        model_name = run_dir.name
                    
                    # Ensure unique names if multiple runs of same model
                    data['model_display_name'] = f"{model_name}\n({run_dir.name})"
                    results.append(data)
    
    # Sort results by run_timestamp to keep chronological order
    results.sort(key=lambda x: x.get('run_timestamp', ''))
    return results

def create_comparison_plots(results, output_file="comparison_results.png"):
    if not results:
        print("No valid benchmark results found.")
        return

    # Extract data
    model_names = [res['model_display_name'] for res in results]
    
    # Setup subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Benchmark Models Comparison', fontsize=16, fontweight='bold')
    
    # --- Plot 1: Overall Average Score ---
    ax1 = axes[0, 0]
    overall_scores = [res['stats'].get('average_score', 0) for res in results]
    bars1 = ax1.bar(model_names, overall_scores, color='skyblue')
    ax1.set_title('Overall Average Score')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_ylim(0, max([1.0] + overall_scores) * 1.1)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')
    ax1.tick_params(axis='x', rotation=45)
    
    # --- Plot 2: Average Score by Category ---
    ax2 = axes[0, 1]
    # Find all unique categories across all runs
    categories = set()
    for res in results:
        categories.update(res.get('stats_by_category', {}).keys())
    categories = sorted(list(categories))
    
    x = np.arange(len(model_names))
    width = 0.8 / len(categories) if categories else 0.8
    
    for i, category in enumerate(categories):
        cat_scores = []
        for res in results:
            cat_stats = res.get('stats_by_category', {}).get(category, {})
            cat_scores.append(cat_stats.get('average', 0))
        
        offset = (i - len(categories)/2 + 0.5) * width
        ax2.bar(x + offset, cat_scores, width, label=category)
        
    ax2.set_title('Average Score by Category')
    ax2.set_ylabel('Score (0-1)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    
    # --- Plot 3: Correctness Distribution (Full/Partial/Wrong) ---
    ax3 = axes[1, 0]
    full_correct = np.array([res['stats'].get('full_correct', 0) for res in results])
    partial_correct = np.array([res['stats'].get('partial_correct', 0) for res in results])
    wrong = np.array([res['stats'].get('wrong', 0) for res in results])
    total = np.array([res['stats'].get('total_questions', 1) for res in results]) # avoid div by zero
    
    # Normalize to percentages
    full_pct = full_correct / total * 100
    partial_pct = partial_correct / total * 100
    wrong_pct = wrong / total * 100
    
    ax3.bar(model_names, full_pct, label='Full Correct', color='green')
    ax3.bar(model_names, partial_pct, bottom=full_pct, label='Partial Correct', color='orange')
    ax3.bar(model_names, wrong_pct, bottom=full_pct + partial_pct, label='Wrong', color='red')
    
    ax3.set_title('Answer Distribution (%)')
    ax3.set_ylabel('Percentage (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    
    # --- Plot 4: Speed (Duration per question) ---
    ax4 = axes[1, 1]
    durations = [res.get('duration_seconds', 0) for res in results]
    answered = [res['stats'].get('answered', 1) for res in results]
    speed = [d / a if a > 0 else 0 for d, a in zip(durations, answered)]
    
    bars4 = ax4.bar(model_names, speed, color='purple')
    ax4.set_title('Average generation time per question')
    ax4.set_ylabel('Seconds / Question')
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.1f}s', ha='center', va='bottom')
    ax4.tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    results = load_results(results_dir)
    print(f"Loaded {len(results)} benchmark runs for comparison.")
    create_comparison_plots(results, output_file=str(Path(__file__).parent / "comparison_plot.png"))
