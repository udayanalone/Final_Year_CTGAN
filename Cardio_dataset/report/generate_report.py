import json
import os
from glob import glob
from datetime import datetime

BASE_DIR = os.path.join('old_tech', 'output')
REPORT_DIR = 'report'

os.makedirs(REPORT_DIR, exist_ok=True)


def collect_metrics():
    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'models': []
    }
    for model_name in ['ctgan', 'dp_ctgan', 'pate_ctgan']:
        model_dir = os.path.join(BASE_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue
        run_dirs = sorted([d for d in glob(os.path.join(model_dir, '*')) if os.path.isdir(d)])
        for run in run_dirs:
            metrics_path = os.path.join(run, 'metrics.json')
            result_path = os.path.join(run, 'result.json')
            entry = {'model': model_name, 'run_dir': run}
            if os.path.isfile(metrics_path):
                try:
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        entry.update(json.load(f))
                except Exception:
                    entry['metrics_error'] = 'Failed to parse metrics.json'
            if os.path.isfile(result_path):
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        entry['result'] = json.load(f)
                except Exception:
                    entry['result_error'] = 'Failed to parse result.json'
            summary['models'].append(entry)
    return summary


def main():
    summary = collect_metrics()
    out_path = os.path.join(REPORT_DIR, 'summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Report written to {out_path}')


if __name__ == '__main__':
    main()
