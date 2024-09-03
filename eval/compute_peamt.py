import os
import csv
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
from datetime import datetime

dt = datetime.now().strftime("%y%m%d_%H%M")

from peamt import *
from peamt.peamt import PEAMT

# paths
project_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
data_path = os.path.join(project_path, "data")
results_path = os.path.join(project_path, "results")
EVAL_DATA_PATHS = {
    "maestro_subset": {
        "meta": os.path.join(data_path, "meta_csv/meta_maestro_subset.csv"),
        "data_path": os.path.join(data_path, "maestro_subset"),
    },
    "revnoise": {
        "meta": os.path.join(data_path, "meta_csv/meta_revnoise_subset.csv"),
        "data_path": os.path.join(data_path, "revnoise"),
    },
}



# set up metric evaluation
eval = PEAMT()

REVERB_IRF = {0: 'norev', 1: 's', 2: 'm', 3: 'l'}
SNR_LEVELS = {0: 'nonoise', 1: 24, 2: 12, 3: 6}  # [dB]


def compute_piece_peamt(subset, piece_path, composer, title, split, piece_id=None, audio=None):
    """
    """
    if subset != 'revnoise':
        header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording', 'peamt']
        results_csv = os.path.join(results_path, f'peamt_{dt}.csv')
    else:
        header = ['composer', 'title', 'split',
                  'model', 'recording', 'peamt', 'reverb', 'snr']
        results_csv = os.path.join(results_path, f'{subset}_peamt_{dt}.csv')

    if os.path.exists(results_csv):
        mode = 'a'
    else:
        mode = 'w'

    if subset != 'revnoise':

        files = sorted(os.listdir(piece_path))
        performers = [f.split('.')[0] for f in files if f.endswith('.match')]

        with open(results_csv, mode) as f:
            csvwriter = csv.writer(f)
            if mode == 'w':
                csvwriter.writerow(header)

            for pidx, performer in enumerate(performers):
                print(
                    f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
                gt_midi = os.path.join(piece_path, f'{performer}.mid')

                if subset == 'maestro_subset':
                    oa_transcriptions = os.path.join(
                        piece_path, f'{performer}_maestro')
                    disklavier_transcriptions = os.path.join(
                        piece_path, f'{performer}_disklavier')
                else:
                    oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                    disklavier_transcriptions = os.path.join(
                        piece_path, 'batik_disklavier')

                for rec_env, rec_tr in zip(['maestro', 'disklavier'],
                                           [oa_transcriptions, disklavier_transcriptions]):
                    print(f' --- {rec_env} audio transcriptions:')
                    for tr_mid in tqdm(os.listdir(rec_tr)):
                        if not tr_mid.endswith('.mid'):
                            continue

                        tr_midi = os.path.join(rec_tr, tr_mid)
                        peamt = eval.evaluate_from_midi(gt_midi, tr_midi)
                        model = tr_mid.split('.')[0].split('_')[0]
                        res = [composer, title, split,
                               performer, model, rec_env, peamt]
                        res = [np.round(r, 4) if isinstance(
                            r, float) else r for r in res]
                        csvwriter.writerow(res)

    else:

        files = [f for f in sorted(os.listdir(piece_path)) if f.startswith(
            str(piece_id)) and f.endswith('.mid')]  # exclude the GT trancsriptions
        gt_midi = [f for f in files if f.startswith(str(piece_id) + 'xx')][0]
        files.remove(gt_midi)  # 45 files : 15 (rev,n) combinations x 3 models
        gt_midi = os.path.join(piece_path, gt_midi)

        with open(results_csv, mode) as f:
            csvwriter = csv.writer(f)
            if mode == 'w':
                csvwriter.writerow(header)

            for i, tr_mid in tqdm(enumerate(files)):
                print(f' --- {i+1:2d}/{len(files)}')
                tr_midi = os.path.join(piece_path, tr_mid)
                peamt = eval.evaluate_from_midi(gt_midi, tr_midi)

                model = tr_mid.split('.')[0].split('_')[0]
                reverb = REVERB_IRF[int(tr_mid.split('_')[0][1])]
                snr = SNR_LEVELS[int(tr_mid.split('_')[0][2])]

                res = [composer, title, split,
                       model, audio, peamt, reverb, snr]
                res = [np.round(r, 4) if isinstance(
                    r, float) else r for r in res]
                res.extend([reverb, snr])
                csvwriter.writerow(res)

    return None


def compute_subset_metrics(subset, subset_meta_csv, eval_set_path):
    """
    Compute metrics for a given subset of the eval data
    Args:
        subset [str] -- name of the subset
        subset_meta_csv [str] -- path to the meta csv for the subset
        eval_set_path [str] -- path to the eval data
    """
    # read the meta csv
    if not isinstance(subset_meta_csv, pd.DataFrame):
        subset_meta_csv = pd.read_csv(subset_meta_csv)
  
    if subset == "maestro_subset":
        grouping_var = ["split", "folder"]

        for i, (_, piece_data) in tqdm(enumerate(subset_meta_csv.groupby(grouping_var))):

            piece_path = os.path.join(
                eval_set_path,
                piece_data["split"].unique()[0],
                piece_data["folder"].unique()[0],
            )
            composer = piece_data["composer"].unique()[0]
            title = piece_data["title"].unique()[0]
            split = piece_data["split"].unique()[0]

            print(69 * "-")
            print(
                f"Computing PEAMT metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")

            compute_piece_peamt(
                subset, piece_path, composer, title, split)

    elif subset == "revnoise":
        for _, row in tqdm(subset_meta_csv.iterrows()):
            print(69 * "-")
            print(f'Computing PEAMT for {row["title"]}')
            compute_piece_peamt(
                subset, EVAL_DATA_PATHS[subset]["data_path"], row["composer"], row["title"], row["split"], row["piece_id"], row["audio_env"]
            )
            
########################################
# compute peamt metric
########################################

parser = argparse.ArgumentParser(description='Compute peamt metric')
parser.add_argument(
    "--subset",
    type=str,
    choices=["maestro_subset", "revnoise"],
    default="maestro_subset",
    help="choose eval data subset",
)

args = parser.parse_args()

subset_meta_csv = EVAL_DATA_PATHS[args.subset]["meta"]
eval_set_path = EVAL_DATA_PATHS[args.subset]["data_path"]

compute_subset_metrics(args.subset, subset_meta_csv, eval_set_path)
