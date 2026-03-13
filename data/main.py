import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 28
})

def get_dir_size(start_path='.'):
    total_size = 0
    if not os.path.exists(start_path):
        return 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def analyze_data():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_md_dir = os.path.join(data_dir, 'raw_md')
    cleaned_md_dir = os.path.join(data_dir, 'cleaned_md')
    qna_path = os.path.join(data_dir, 'llm_qna.jsonl')
    removed_path = os.path.join(data_dir, 'llm_qna_removed.json')

    # 1. Webscraping stats
    raw_files = os.listdir(raw_md_dir) if os.path.exists(raw_md_dir) else []
    cleaned_files = os.listdir(cleaned_md_dir) if os.path.exists(cleaned_md_dir) else []
    
    num_webscraped = len(raw_files)
    num_after_cleaning = len(cleaned_files)
    
    html_raw_size = 593.4  # Hardcoded as requested
    raw_size = get_dir_size(raw_md_dir) / (1024 * 1024) # MB
    cleaned_size = get_dir_size(cleaned_md_dir) / (1024 * 1024) # MB
    reduction_pct = (1 - (cleaned_size / html_raw_size)) * 100 if html_raw_size > 0 else 0

    print(f"--- General Stats ---")
    print(f"Files webscraped: {num_webscraped}")
    print(f"Files after cleaning: {num_after_cleaning}")
    print(f"HTML Raw size: {html_raw_size:.2f} MB")
    print(f"MD Raw size: {raw_size:.2f} MB")
    print(f"MD Cleaned size: {cleaned_size:.2f} MB")
    print(f"Total size reduction: {reduction_pct:.2f}%")

    # 2. QnA Analysis
    qna_data = []
    if os.path.exists(qna_path):
        with open(qna_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        qna_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    df_qna = pd.DataFrame(qna_data)
    
    # Filter out 'Other' category as requested
    if not df_qna.empty and 'category' in df_qna.columns:
        df_qna = df_qna[df_qna['category'] != 'Other']
    
    # Category distribution
    cat_counts = {}
    if not df_qna.empty and 'category' in df_qna.columns:
        cat_counts = df_qna['category'].value_counts().to_dict()
        print(f"\n--- Category Distribution (Questions) ---")
        for cat, count in cat_counts.items():
            print(f"{cat}: {count}")

    # Relevancy analysis based on Markdown files
    # A file is considered "relevant" if it has questions in llm_qna.jsonl
    # A file is considered "irrelevant" if it doesn't have questions (often marked in index.json or just absent)
    # We'll use the hashes present in llm_qna.jsonl vs total files in cleaned_md
    relevant_hashes = set(df_qna['hash'].unique()) if not df_qna.empty else set()
    num_relevant_docs = len(relevant_hashes)
    num_irrelevant_docs = num_after_cleaning - num_relevant_docs
    
    print(f"\n--- Relevancy (Markdown Files) ---")
    print(f"Relevant documents: {num_relevant_docs}")
    print(f"Irrelevant documents: {num_irrelevant_docs}")

    # 3. Questions per document
    avg_q, min_q, max_q = 0, 0, 0
    if not df_qna.empty and 'hash' in df_qna.columns:
        q_per_doc = df_qna.groupby('hash').size()
        avg_q = q_per_doc.mean()
        min_q = q_per_doc.min()
        max_q = q_per_doc.max()
        print(f"\n--- Questions per Document (of relevant ones) ---")
        print(f"Average: {avg_q:.2f}")
        print(f"Min: {min_q}")
        print(f"Max: {max_q}")

    # 4. Generate metadata.json
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "processing_info": {
            "hardware": "Mac Mini M4",
            "model": "gpt-oss-20b",
            "total_duration_hours": round(67984.26 / 3600, 2),
            "total_duration_seconds": 67984.26,
            "tqdm_snapshot": "Processing Documents: 100%|█| 3092/3092 [18:53:04<00:00, 21.99s/doc, failed=2, hash=1fa1d58c, irrelevant=1425, questions=13555]"
        },
        "scraping": {
            "total_files_scraped": num_webscraped,
            "files_after_cleaning": num_after_cleaning,
            "raw_size_mb": round(raw_size, 2),
            "cleaned_size_mb": round(cleaned_size, 2),
            "reduction_percentage": round(reduction_pct, 2)
        },
        "document_relevancy": {
            "relevant_docs": num_relevant_docs,
            "irrelevant_docs": num_irrelevant_docs,
            "total_docs": num_after_cleaning
        },
        "llm_stats": {
            "total_questions": len(qna_data),
            "categories": cat_counts,
            "questions_per_doc": {
                "avg": round(float(avg_q), 2),
                "min": int(min_q),
                "max": int(max_q)
            }
        }
    }
    
    with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\nMetadata saved to metadata.json")
    
    print(f"\n--- LLM Processing Info ---")
    print(f"Hardware: Mac Mini M4")
    print(f"Model: gpt-oss-20b")
    print(f"Duration: {metadata['processing_info']['total_duration_hours']} hours (67984.26 seconds)")
    print(f"Snapshot: {metadata['processing_info']['tqdm_snapshot']}")

    # 5. Plots
    plt.figure(figsize=(15, 12))

    # Plot 1: Size Reduction
    plt.subplot(2, 2, 1)
    sizes = [html_raw_size, raw_size, cleaned_size]
    labels = ['Raw HTML', 'Raw MD', 'Cleaned MD']
    ax1 = sns.barplot(x=labels, y=sizes, palette='viridis')
    # Add labels above the bars for better visibility
    for i, size in enumerate(sizes):
        ax1.text(i, size + (max(sizes) * 0.01), f'{size:.2f} MB', ha='center', color='black', fontweight='bold')
    plt.title('Data Pipeline Size Reduction')
    plt.ylabel('Size in MB')

    # Plot 2: Category Distribution
    plt.subplot(2, 2, 2)
    if not df_qna.empty and 'category' in df_qna.columns:
        cat_order = df_qna['category'].value_counts().index
        ax2 = sns.countplot(data=df_qna, y='category', order=cat_order, palette='magma')
        # Add labels inside the bars
        for i, p in enumerate(ax2.patches):
            width = p.get_width()
            if width > 0:
                ax2.text(width / 2, p.get_y() + p.get_height() / 2, f'{int(width)}', 
                        ha='center', va='center', color='white', fontweight='bold')
        plt.title('Questions per Category')
    else:
        plt.text(0.5, 0.5, 'No QnA data', ha='center')

    # Plot 3: Questions per Doc (Distribution)
    plt.subplot(2, 2, 3)
    if not df_qna.empty and 'hash' in df_qna.columns:
        sns.histplot(q_per_doc, bins=range(0, int(max_q) + 2), kde=False, color='skyblue')
        plt.title('Distribution of Questions per Doc (Relevant Only)')
        plt.xlabel('Number of Questions')
    else:
        plt.text(0.5, 0.5, 'No QnA data', ha='center')

    # Plot 4: Document Relevancy
    plt.subplot(2, 2, 4)
    rel_labels = ['Relevant Docs', 'Irrelevant Docs']
    rel_counts = [num_relevant_docs, num_irrelevant_docs]
    if sum(rel_counts) > 0:
        plt.pie(rel_counts, labels=rel_labels, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
        plt.title('File Relevancy (Chosen by LLM)')
    else:
        plt.text(0.5, 0.5, 'No docs found', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'data_analysis.png'))
    print(f"Plots saved to data_analysis.png")

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'data_analysis.png'))
    print(f"Plots saved to data_analysis.png")

if __name__ == "__main__":
    analyze_data()
