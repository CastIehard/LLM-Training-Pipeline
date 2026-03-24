import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


NUM_RECENT_BENCHMARKS = 2
TOP_K_PER_GROUP = 10


def clean_display_name(name):
    prefix = "Qwen_Qwen3-0.6B"
    if not isinstance(name, str):
        return name

    # Keep the prefix only for the base model case where it is followed by a space
    if name.startswith(prefix + " "):
        return name

    if name.startswith(prefix):
        name = name[len(prefix):]
        while name.startswith((" ", "_", "-", "/")):
            name = name[1:]

    return name


def is_sft_model(name):
    if not isinstance(name, str):
        return False
    return "-SFT" in name


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

                    clean_model_name = clean_display_name(model_name)
                    clean_run_name = clean_display_name(run_dir.name)

                    data['clean_model_name'] = clean_model_name
                    data['is_sft'] = is_sft_model(clean_model_name)

                    # Ensure unique names if multiple runs of same model
                    data['model_display_name'] = f"{clean_model_name}\n({clean_run_name})"
                    results.append(data)

    # Sort results by run_timestamp to keep chronological order
    results.sort(key=lambda x: x.get('run_timestamp', ''))
    return results


def select_results_for_plot(results, num_recent=NUM_RECENT_BENCHMARKS, top_k_per_group=TOP_K_PER_GROUP):
    if not results:
        return results

    selected_indices = set()

    # Always include the very first benchmark
    selected_indices.add(0)

    # Include the last N benchmarks
    recent_start = max(0, len(results) - num_recent)
    selected_indices.update(range(recent_start, len(results)))

    non_sft_indices = [i for i, res in enumerate(results) if not res.get('is_sft', False)]
    sft_indices = [i for i, res in enumerate(results) if res.get('is_sft', False)]

    def add_top_k(indices):
        ranked = sorted(
            indices,
            key=lambda i: results[i].get('stats', {}).get('average_score', 0),
            reverse=True
        )
        selected_indices.update(ranked[:top_k_per_group])

    # Include top-k overall for base and SFT separately
    add_top_k(non_sft_indices)
    add_top_k(sft_indices)

    # Preserve chronological order before later grouping
    return [results[i] for i in range(len(results)) if i in selected_indices]


def reorder_results_with_sft_grouping(results):
    non_sft = [res for res in results if not res.get('is_sft', False)]
    sft = [res for res in results if res.get('is_sft', False)]
    return non_sft + sft, len(non_sft), len(sft)


def get_grouped_x_positions(num_non_sft, num_sft, gap=1.2):
    x_positions = []
    for i in range(num_non_sft):
        x_positions.append(i)
    for i in range(num_sft):
        x_positions.append(num_non_sft + gap + i)
    return np.array(x_positions, dtype=float)


def add_sft_divider(ax, x_positions, num_non_sft, num_sft):
    if num_non_sft > 0 and num_sft > 0:
        divider_x = (x_positions[num_non_sft - 1] + x_positions[num_non_sft]) / 2
        ax.axvline(divider_x, color='gray', linestyle='--', alpha=0.5)


def create_comparison_plots(results, output_file="comparison_results.png"):
    if not results:
        print("No valid benchmark results found.")
        return

    results = select_results_for_plot(results)
    results, num_non_sft, num_sft = reorder_results_with_sft_grouping(results)

    # Extract data
    model_names = [res['model_display_name'] for res in results]
    x = get_grouped_x_positions(num_non_sft, num_sft)

    # Setup subplots with larger figure and a double-height answer distribution plot
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.35, wspace=0.25)
    fig.suptitle('Benchmark Models Comparison', fontsize=20, fontweight='bold')

    # --- Plot 1: Overall Average Score ---
    ax1 = fig.add_subplot(gs[0, 0])
    overall_scores = [res['stats'].get('average_score', 0) for res in results]
    bars1 = ax1.bar(x, overall_scores, color='skyblue')
    ax1.set_title('Overall Average Score')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=35, ha='right')
    ax1.set_ylim(0, max([1.0] + overall_scores) * 1.1)
    add_sft_divider(ax1, x, num_non_sft, num_sft)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')

    # --- Plot 2: Average Score by Category ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Find all unique categories across all runs
    categories = set()
    for res in results:
        categories.update(res.get('stats_by_category', {}).keys())
    categories = sorted(list(categories))

    width = 0.8 / len(categories) if categories else 0.8

    for i, category in enumerate(categories):
        cat_scores = []
        for res in results:
            cat_stats = res.get('stats_by_category', {}).get(category, {})
            cat_scores.append(cat_stats.get('average', 0))

        offset = (i - len(categories) / 2 + 0.5) * width
        ax2.bar(x + offset, cat_scores, width, label=category)

    ax2.set_title('Average Score by Category')
    ax2.set_ylabel('Score (0-1)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=35, ha='right')
    add_sft_divider(ax2, x, num_non_sft, num_sft)
    ax2.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')

    # --- Plot 3: Correctness Distribution (Full/Partial/Wrong) ---
    ax3 = fig.add_subplot(gs[1, :])
    full_correct = np.array([res['stats'].get('full_correct', 0) for res in results])
    partial_correct = np.array([res['stats'].get('partial_correct', 0) for res in results])
    wrong = np.array([res['stats'].get('wrong', 0) for res in results])
    total = np.array([res['stats'].get('total_questions', 1) for res in results])

    # Normalize to percentages
    full_pct = np.where(total > 0, full_correct / total * 100, 0)
    partial_pct = np.where(total > 0, partial_correct / total * 100, 0)
    wrong_pct = np.where(total > 0, wrong / total * 100, 0)

    ax3.bar(x, full_pct, label='Full Correct', color='green')
    ax3.bar(x, partial_pct, bottom=full_pct, label='Partial Correct', color='orange')
    ax3.bar(x, wrong_pct, bottom=full_pct + partial_pct, label='Wrong', color='red')

    ax3.set_title('Answer Distribution (%)')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=35, ha='right')
    add_sft_divider(ax3, x, num_non_sft, num_sft)
    ax3.legend(ncol=3, loc='upper center')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    results_dir = Path(__file__).parent / "results"
    results = load_results(results_dir)
    print(f"Loaded {len(results)} benchmark runs for comparison.")
    create_comparison_plots(results, output_file=str(Path(__file__).parent / "comparison_plot.png"))