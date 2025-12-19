#!/usr/bin/env python3
"""Pareto frontier analysis: accuracy vs GPU-hours.

Pareto-optimal means no other model is both *faster* (lower GPU-hours) and
*more accurate*.

Inputs:
    - `results/*.log` JSON files containing `duration_s`, `accuracy`, and optionally `num_gpus`.

Outputs (by default in the current working directory):
    - `pareto_en.png`
    - `pareto_no.png`
"""

from __future__ import annotations

import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

try:
        from adjustText import adjust_text  # type: ignore
except Exception:  # pragma: no cover
        adjust_text = None

# model family colors - high contrast, distinguishable
FAMILY_COLORS = {
    'deepseek': '#e41a1c',    # bright red
    'llama': '#377eb8',       # blue
    'qwen': '#4daf4a',        # green
    'gemma': '#ff7f00',       # orange
    'magistral': '#984ea3',   # purple
    'norwai': '#00CED1',      # dark turquoise
    'mimir': '#8B4513',       # saddle brown
    'gpt-oss': '#f781bf',     # pink
    'qwq': '#a65628',         # brown
}

# parameter counts (in billions)
PARAM_COUNTS = {
    'deepseek-r1:1.5b': 1.5, 'deepseek-r1:7b': 7, 'deepseek-r1:14b': 14,
    'deepseek-r1:32b': 32, 'deepseek-r1:70b': 70, 'deepseek-r1:671b': 671,
    'llama3.1:8b': 8, 'llama3.1:70b': 70, 'llama3.1:405b': 405,
    'llama3.1:70b-instruct-q4': 70,
    'qwen2.5:0.5b': 0.5, 'qwen2.5:3b': 3, 'qwen2.5:7b': 7, 'qwen2.5:14b': 14,
    'qwen2.5:32b': 32, 'qwen2.5:72b': 72, 'qwen2.5:230b': 230,
    'qwen3:0.6b': 0.6, 'qwen3:1.7b': 1.7, 'qwen3:8b': 8, 'qwen3:14b': 14,
    'qwen3:32b': 32, 'qwen3:235b': 235,
    'qwq:32b': 32,
    'gemma3:270m': 0.27, 'gemma3:1b': 1, 'gemma3:4b': 4, 'gemma3:27b': 27,
    'magistral:24b': 24, 'magistral:24b-en': 24, 'magistral:24b-no': 24,
    'NorwAI-Magistral-24B-reasoning:Q8:0': 24,
    'norwai-mixtral-8x7b': 47, 'norwai-mixtral-8x7b:en': 47, 'norwai-mixtral-8x7b:no': 47,
    'mimir-mistral-7b-core-instruct-Q4': 7, 'mimir-mistral:7b-core-instruct-Q4': 7,
    'mimir-mistral-7b-core-scratch-instruct-Q4': 7, 'mimir-mistral:7b-core-scratch-instruct-Q4': 7,
    'gpt-oss:20b': 20, 'gpt-oss:120b': 120,
}

def get_model_family(model_name):
    """extract model family from name"""
    name_lower = model_name.lower()
    for family in FAMILY_COLORS.keys():
        if family in name_lower:
            return family
    return 'other'

def load_log(log_path):
    """get runtime and accuracy from log file"""
    data = json.loads(Path(log_path).read_text(encoding="utf-8"))

    # Remove language suffix if present (supports `_en`, `-en`, `:en` variants).
    stem = Path(log_path).stem
    model_base = stem
    for lang in ("en", "no"):
        for sep in ("_", "-", ":"):
            suffix = f"{sep}{lang}"
            if stem.endswith(suffix) and len(stem) > len(suffix):
                model_base = stem[: -len(suffix)]
                break
        else:
            continue
        break
    params = PARAM_COUNTS.get(model_base, 0)
    num_gpus = data.get('num_gpus', 1)
    
    return {
        'model': log_path.stem,
        'model_base': model_base,
        'hours': data['duration_s'] / 3600,
        'gpu_hours': (data['duration_s'] / 3600) * num_gpus,
        'accuracy': data['accuracy'] * 100,
        'params': params,
        'num_gpus': num_gpus,
        'family': get_model_family(log_path.stem)
    }

def is_pareto_optimal(models):
    """
    find models where no other model beats them in BOTH dimensions
    a model is optimal if no other has BOTH less gpu time AND higher accuracy
    """
    pareto = []
    for i, m1 in enumerate(models):
        dominated = False
        for j, m2 in enumerate(models):
            if i != j:
                # m2 dominates m1 if it uses less gpu time AND has higher accuracy
                if m2['gpu_hours'] < m1['gpu_hours'] and m2['accuracy'] > m1['accuracy']:
                    dominated = True
                    break
        pareto.append(not dominated)
    return pareto


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Pareto frontier for accuracy vs GPU-hours")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing .log files")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory to write PNGs")
    parser.add_argument("--dpi", type=int, default=200)
    return parser

def main() -> int:
    args = build_argparser().parse_args()
    results_dir: Path = args.results_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # load all log files
    all_models = []
    
    for log_file in sorted(results_dir.glob('*.log')):
        try:
            all_models.append(load_log(log_file))
        except Exception as e:
            print(f"skip {log_file.name}: {e}")
    
    print(f"\nloaded {len(all_models)} model runs")
    
    # separate by language
    def _has_lang_suffix(name: str, lang: str) -> bool:
        return any(name.endswith(f"{sep}{lang}") for sep in ("_", "-", ":"))

    en_models = [m for m in all_models if _has_lang_suffix(m['model'], 'en')]
    no_models = [m for m in all_models if _has_lang_suffix(m['model'], 'no')]
    
    print(f"english models: {len(en_models)}")
    print(f"norwegian models: {len(no_models)}")
    
    total_gpu_hours = sum(m['gpu_hours'] for m in all_models)
    print(f"total gpu hours: {total_gpu_hours:.1f}h")
    
    # create plots for both languages
    for lang, models in [('en', en_models), ('no', no_models)]:
        if not models:
            continue
        
        print(f"\nprocessing {lang} models...")
        create_pareto_plot(models, lang, total_gpu_hours, out_dir=out_dir, dpi=int(args.dpi))
    return 0

def create_pareto_plot(models: list[dict[str, Any]], lang: str, total_gpu_hours: float, *, out_dir: Path, dpi: int) -> None:
    """create pareto plot for a specific language"""
    # find pareto frontier
    pareto = is_pareto_optimal(models)
    
    # print results sorted by gpu hours
    print(f"\npareto optimal models ({lang}):")
    for m, p in sorted(zip(models, pareto), key=lambda x: x[0]['gpu_hours']):
        if p:
            print(f"  {m['model_base']}: {m['params']:.1f}B params, {m['gpu_hours']:.2f} gpu-h ({m['num_gpus']} gpu × {m['hours']:.2f}h), {m['accuracy']:.1f}% acc")
    
    # make plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # plot all points
    texts = []
    for m, p in zip(models, pareto):
        color = FAMILY_COLORS.get(m['family'], '#95a5a6')
        marker_size = 150 if p else 80
        alpha = 0.9 if p else 0.4
        ax.scatter(m['gpu_hours'], m['accuracy'], c=color, s=marker_size, alpha=alpha,
                  edgecolors='black', linewidths=2 if p else 0.5, zorder=3 if p else 2)
    
    # add labels for pareto optimal points (non-overlapping)
    for m, p in zip(models, pareto):
        if p:
            # shorter label: model_base without _en/_no
            label = m['model_base'].split(':')[0] + ':' + m['model_base'].split(':')[-1]
            label += f"\n{m['params']:.0f}B" if m['params'] >= 1 else f"\n{m['params']*1000:.0f}M"
            txt = ax.text(m['gpu_hours'], m['accuracy'], label,
                         fontsize=8, ha='center', va='bottom', 
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor='gray', alpha=0.8))
            texts.append(txt)
    
    # adjust text positions to avoid overlap (optional dependency)
    if adjust_text is not None:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    # connect pareto frontier with line
    pareto_models = sorted([m for m, p in zip(models, pareto) if p], 
                          key=lambda x: x['gpu_hours'])
    if len(pareto_models) > 1:
        gpu_hours_line = [m['gpu_hours'] for m in pareto_models]
        acc_line = [m['accuracy'] for m in pareto_models]
        ax.plot(gpu_hours_line, acc_line, 'k--', alpha=0.4, linewidth=2, zorder=1)
    
    # better labels
    lang_name = 'English' if lang == 'en' else 'Norwegian'
    ax.set_xlabel('total gpu hours (runtime × num_gpus)', fontsize=13, weight='bold')
    ax.set_ylabel('accuracy (%)', fontsize=13, weight='bold')
    
    title = f'pareto frontier: accuracy vs gpu hours ({lang_name})\n'
    title += f'(larger points = pareto optimal | total gpu hours across all runs: {total_gpu_hours:.1f}h)'
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # legend for model families
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=FAMILY_COLORS[fam], edgecolor='black', label=fam.capitalize())
                      for fam in sorted(set(m['family'] for m in models))
                      if fam in FAMILY_COLORS]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
             framealpha=0.95, title='Model Family', title_fontsize=12)
    
    plt.tight_layout()
    filename = out_dir / f'pareto_{lang}.png'
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"saved {filename}")
    plt.close()

if __name__ == "__main__":
    raise SystemExit(main())
