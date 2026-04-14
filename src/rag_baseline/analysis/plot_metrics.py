import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAG Baseline Analysis Plots")
    parser.add_argument("--output-dir", "-i", type=str, default="outputs",
                        help="Directory containing baseline output folders (default: outputs)")
    parser.add_argument("--plots-dir", "-o", type=str, default="analysis_plots",
                        help="Directory to save generated plots (default: analysis_plots)")
    return parser.parse_args()

def aggregate_metrics(output_dir: Path) -> pd.DataFrame:
    """Reads all summary_metrics.json and extracts them into a pandas DataFrame."""
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist.")
        sys.exit(1)
        
    data = []
    # Recalculate everything from evaluations.jsonl if it exists
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        # Skip smoke test directories automatically
        if "smoke_test" in run_dir.name:
            continue
            
        summary_file = run_dir / "summary_metrics.json"
        
        # Determine the name of the folder for reference 
        run_name = run_dir.name
        
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    metrics = json.load(f)
                    
                    # Store folder name as fall back baseline name
                    if 'baseline_name' not in metrics:
                        metrics['baseline_name'] = run_name
                        
                    # Calculate Unanswerable/Conflict Rates and Recalculate missing basic metrics
                    evals_file = run_dir / "evaluations.jsonl"
                    if evals_file.exists():
                         _augment_with_dataset_specific_metrics(evals_file, metrics)

                    data.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to read metrics from {summary_file}. Error: {e}")
                
    if not data:
        print("Note: No summary_metrics.json files found.")
        sys.exit(0)
        
    return pd.DataFrame(data)

def _augment_with_dataset_specific_metrics(evals_file: Path, metrics: dict):
    """Extract fine-grained metrics for dataset analysis."""
    import jsonlines
    
    predictions = []
    try:
         with jsonlines.open(evals_file) as reader:
             for obj in reader:
                 predictions.append(obj)
    except Exception as e:
         print(f"Warning: Could not read {evals_file}: {e}")
         return
         
    metrics['total_eval_count'] = len(predictions)
    
    # Recalculate true metrics from evaluations.jsonl since summary_metrics might be incomplete
    valid_em = [p.get('metrics', {}).get('exact_match') for p in predictions if p.get('metrics', {}).get('exact_match') is not None]
    if valid_em:
        metrics['exact_match_rate'] = sum(valid_em) / len(valid_em)
        
    valid_nm = [p.get('metrics', {}).get('normalized_match') for p in predictions if p.get('metrics', {}).get('normalized_match') is not None]
    if valid_nm:
        metrics['normalized_match_rate'] = sum(valid_nm) / len(valid_nm)
        
    valid_ma = [p.get('metrics', {}).get('multi_answer_score') for p in predictions if p.get('metrics', {}).get('multi_answer_score') is not None]
    if valid_ma:
        metrics['multi_answer_score'] = sum(valid_ma) / len(valid_ma)
        
    # Categorize AmbigDocs and RAMDocs Errors
    if metrics.get('dataset') in ['ambigdocs', 'ramdocs']:
        # 'wrong'   — model answered but covered zero gold answers (confident hallucination)
        # 'no_answer' — model abstained (empty / unknown output)
        # These two are intentionally kept separate: conflating them hides whether
        # the model is hallucinating vs. appropriately abstaining.
        categories = ['complete', 'partial', 'ambiguous', 'merged', 'wrong', 'no_answer']
        for cat in categories:
            metrics[f'error_category_{cat}'] = sum(1 for p in predictions if p.get('metrics', {}).get('answer_category') == cat)
    
    # For faith eval specific tasks
    if metrics.get('dataset') == 'faitheval':
        # E.g. Check for unanswerable outputs
        unknown_count = sum(1 for p in predictions if p.get('metrics', {}).get('exact_match', False) and 'un' in p.get('example_id', ''))
        metrics['unanswerable_em_count'] = unknown_count
        
        conflict_count = sum(1 for p in predictions if p.get('metrics', {}).get('exact_match', False) and 'ic' in p.get('example_id', ''))
        metrics['inconsistent_em_count'] = conflict_count


def generate_accuracy_plots(df: pd.DataFrame, plots_dir: Path):
    """Generates bar plots for EM and Normalized Match rates per dataset."""
    # Ensure standard numeric format
    df['exact_match_rate'] = pd.to_numeric(df.get('exact_match_rate', 0), errors='coerce')
    df['normalized_match_rate'] = pd.to_numeric(df.get('normalized_match_rate', 0), errors='coerce')
    
    dataset_list = df['dataset'].dropna().unique()
    sns.set_theme(style="whitegrid")

    for dataset in dataset_list:
        subset = df[df['dataset'] == dataset].copy()
        
        # Setup plot size and visual aspects
        plt.figure(figsize=(10, 6))
        
        melted = subset.melt(
            id_vars=['baseline_name'], 
            value_vars=['exact_match_rate', 'normalized_match_rate'],
            var_name='Metric',
            value_name='Score'
        )
        
        ax = sns.barplot(data=melted, x='baseline_name', y='Score', hue='Metric', palette="viridis")
        
        # Annotate bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        plt.title(f'Pipeline Overall Accuracy ({dataset.upper()})', fontsize=14, pad=15)
        plt.xlabel('Baseline / Pipeline', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.1)  # Since these are rates, bound to 1.1 to leave room for labels
        plt.tight_layout()
        
        save_path = plots_dir / f"{dataset}_accuracy_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

def generate_multi_answer_plots(df: pd.DataFrame, plots_dir: Path):
    """If datasets like AmbigDocs/RAMDocs have multi_answer_score, plot them."""
    if 'multi_answer_score' not in df.columns:
        return
    
    sns.set_theme(style="whitegrid")
    # Subsets that contain Multi Answer Score
    multi_df = df.dropna(subset=['multi_answer_score']).copy()
    
    if multi_df.empty:
         return
         
    dataset_list = multi_df['dataset'].unique()
    
    for dataset in dataset_list:
         subset = multi_df[multi_df['dataset'] == dataset].copy()
         plt.figure(figsize=(10, 6))
         
         melted = subset.melt(
             id_vars=['baseline_name'], 
             value_vars=['multi_answer_score'],
             var_name='Metric',
             value_name='Score'
         )
         
         ax = sns.barplot(data=melted, x='baseline_name', y='Score', hue='Metric', palette="rocket")
         
         for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)
            
         plt.title(f'Multi-Answer Retrieval & Accuracy ({dataset.upper()})', fontsize=14, pad=15)
         plt.xlabel('Baseline / Pipeline', fontsize=12)
         plt.ylabel('Score', fontsize=12)
         plt.xticks(rotation=45, ha="right")
         plt.ylim(0, 1.1)
         plt.tight_layout()
         
         save_path = plots_dir / f"{dataset}_multi_answer.png"
         plt.savefig(save_path, dpi=300)
         plt.close()
         print(f"Saved: {save_path}")

def generate_error_category_plots(df: pd.DataFrame, plots_dir: Path):
    """Plots stacked bar charts for error categories in AmbigDocs/RAMDocs"""
    sns.set_theme(style="whitegrid")
    
    category_cols = [col for col in df.columns if col.startswith('error_category_')]
    if not category_cols:
        return
        
    # Get datasets that have error category data
    target_datasets = df[df[category_cols].notna().any(axis=1)]['dataset'].unique()
    
    for dataset in target_datasets:
        subset = df[df['dataset'] == dataset].copy()
        if subset.empty:
            continue
            
        plt.figure(figsize=(12, 7))
        
        # Normalize by total_eval_count to get percentages
        for col in category_cols:
            subset[f'{col}_pct'] = subset[col] / subset['total_eval_count'] * 100
            
        pct_cols = [f'{col}_pct' for col in category_cols]
        
        # We need a clean dataframe for stacked plotting
        plot_df = subset[['baseline_name'] + pct_cols].set_index('baseline_name')
        
        # Rename columns for the legend (remove 'error_category_' and '_pct')
        plot_df.columns = [col.replace('error_category_', '').replace('_pct', '').title() for col in plot_df.columns]
        
        ax = plot_df.plot(kind='bar', stacked=True, figsize=(12, 7),
                         color={
                             'Complete':  '#2ca02c',   # green  — fully correct
                             'Partial':   '#98df8a',   # light green — partially correct
                             'Ambiguous': '#ff7f0e',   # orange — single answer, missed ambiguity
                             'Merged':    '#c5b0d5',   # lavender — merged multiple golds
                             'Wrong':     '#d62728',   # red    — answered, zero recall (hallucination)
                             'No_Answer': '#aec7e8',   # light blue — abstained / empty output
                         })
        
        plt.title(f'Answer Categories & Confident Hallucinations ({dataset.upper()})', fontsize=14, pad=15)
        plt.xlabel('Pipeline', fontsize=12)
        plt.ylabel('Percentage of Queries (%)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = plots_dir / f"{dataset}_error_categories.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved: {save_path}")

def generate_faitheval_plots(df: pd.DataFrame, plots_dir: Path):
    """Plots robustness rates: unanswerable and inconsistent detection"""
    if 'unanswerable_em_count' not in df.columns:
        return
        
    subset = df[df['dataset'] == 'faitheval'].copy()
    if subset.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    subset['Unanswerable Exact Match (%)'] = subset.get('unanswerable_em_count', 0) / subset.get('total_eval_count', 1)
    subset['Inconsistent Exact Match (%)'] = subset.get('inconsistent_em_count', 0) / subset.get('total_eval_count', 1)
    
    melted = subset.melt(
        id_vars=['baseline_name'],
        value_vars=['Unanswerable Exact Match (%)', 'Inconsistent Exact Match (%)'],
        var_name='Robustness Category',
        value_name='Match Rate'
    )
    
    ax = sns.barplot(data=melted, x='baseline_name', y='Match Rate', hue='Robustness Category', palette="magma")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
        
    plt.title('FaithEval Robustness Analysis', fontsize=14, pad=15)
    plt.xlabel('Pipeline', fontsize=12)
    plt.ylabel('Exact Match Rate on Sub-task', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    
    save_path = plots_dir / "faitheval_robustness.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Aggregating metrics from {output_dir.absolute()} ...")
    df = aggregate_metrics(output_dir)
    
    print(f"Found {len(df)} run configurations.")
    
    # 1. Dataset Accuracy plots (LLM-Only vs RAG vs Reranker)
    generate_accuracy_plots(df, plots_dir)
    
    # 2. Multi-Answer specialized plots
    generate_multi_answer_plots(df, plots_dir)
    
    # 3. Error Category specialized plots (AmbigDocs/RAMDocs)
    generate_error_category_plots(df, plots_dir)
    
    # 4. Faithfulness specialized plots
    generate_faitheval_plots(df, plots_dir)
    
    # Dump the merged dataframe for manual inspection
    csv_path = plots_dir / "aggregated_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw aggregated data to {csv_path}")

if __name__ == "__main__":
    main()
