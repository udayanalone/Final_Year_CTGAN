import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description='Train PATE-CTGAN and generate synthetic data.')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--output_dir', type=str, default=os.path.join('old_tech', 'output', 'pate_ctgan'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--teachers', type=int, default=10, help='Number of teacher models in PATE')
    parser.add_argument('--categorical', type=str, default='', help='Comma-separated categorical columns')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    ensure_dir(args.output_dir)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, run_id)
    ensure_dir(run_dir)

    save_json({
        'data': args.data,
        'epochs': args.epochs,
        'samples': args.samples,
        'teachers': args.teachers,
        'categorical': args.categorical.split(',') if args.categorical else []
    }, os.path.join(run_dir, 'config.json'))

    try:
        import pandas as pd
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
    except Exception as e:
        save_json({'status': 'failed', 'reason': f'Missing dependency or import error: {e}'}, os.path.join(run_dir, 'result.json'))
        print('PATE-CTGAN dependencies missing. Install with: pip install sdv pandas', file=sys.stderr)
        return 1

    # Placeholder for PATE-CTGAN. Implement PATE mechanism or integrate a library if available.
    # Load data with delimiter auto-detection
    df = pd.read_csv(args.data, sep=None, engine='python')

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)

    synthetic = synthesizer.sample(num_rows=args.samples)

    data_path = os.path.join(run_dir, 'synthetic.csv')
    synthetic.to_csv(data_path, index=False)

    # Also save anonymized dataset into Generated_dataset/pate_ctgan/<timestamp>
    generated_dir = os.path.join('Generated_dataset', 'pate_ctgan', run_id)
    ensure_dir(generated_dir)
    generated_path = os.path.join(generated_dir, 'synthetic.csv')
    synthetic.to_csv(generated_path, index=False)

    metrics = {
        'num_real_rows': int(len(df)),
        'num_synth_rows': int(len(synthetic)),
        'columns': list(df.columns),
        'run_id': run_id,
        'model': 'PATE-CTGAN',
        'teachers': args.teachers,
        'note': 'This is a placeholder using CTGANSynthesizer without PATE guarantees.',
        'generated_dataset_path': generated_path
    }
    save_json(metrics, os.path.join(run_dir, 'metrics.json'))
    save_json({'status': 'success', 'run_dir': run_dir}, os.path.join(run_dir, 'result.json'))

    print(f'PATE-CTGAN (placeholder) completed. Outputs saved to: {run_dir} and {generated_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
