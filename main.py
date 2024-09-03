import argparse

from config import *
from helpers.compute_metrics import compute_subset_metrics

parser = argparse.ArgumentParser(
    description="Compute musically informed transcription metrics."
)

parser.add_argument(
    "--subset",
    type=str,
    choices=["maestro_subset", "revnoise"],
    default="maestro_subset",
    help="choose eval data subset",
)
parser.add_argument(
    "--eval_metric",
    type=str,
    choices=[
        "frame",
        "note_offset",
        "note_offset_velocity",
        "dynamics",
        "harmony",
        "articulation",
        "timing"
    ],
    default="all",
    help="choose type of metric to compute",
)
args = parser.parse_args()

subset_meta_csv = EVAL_DATA_PATHS[args.subset]["meta"]
eval_set_path = EVAL_DATA_PATHS[args.subset]["data_path"]

"""
COMPUTE METRICS
"""
if args.eval_metric != 'all':
    print(f'Computing {args.eval_metric} metrics for {args.subset} subset')
    compute_subset_metrics(
        args.subset, subset_meta_csv, eval_set_path, metric_type=args.eval_metric)
else:
    print(f'Computing all metrics for {args.subset} subset')
    for metric in ['frame', 'note_offset', 'note_offset_velocity', 'dynamics', 'harmony', 'articulation', 'timing']:
        compute_subset_metrics(
            args.subset, subset_meta_csv, eval_set_path, metric_type=metric)