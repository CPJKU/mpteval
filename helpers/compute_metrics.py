import os
import csv
import pandas as pd

from eval.metrics.benchmarks import *
from eval.metrics.dynamics import *
from eval.metrics.harmony import *
from eval.metrics.articulation import *
from eval.metrics.timing import *

from config import results

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import datetime

dt = datetime.now().strftime("%y%m%d_%H%M")

REVERB_IRF = {0: 'norev', 1: 's', 2: 'm', 3: 'l'}
SNR_LEVELS = {0: 'nonoise', 1: 24, 2: 12, 3: 6}  # [dB]


def construct_results_filepath(metric, subset):
    if metric in ['frame', 'note_offset', 'note_offset_velocity']:
        results_csv = os.path.join(results, f'IR_{metric}_{dt}.csv')
    else:
        results_csv = os.path.join(results, f'musical_{metric}_{dt}.csv')
    if subset == 'revnoise':
        results_csv_fn = 'revnoise_' + results_csv.split('/')[-1].split('.')[0]
        results_csv = os.path.join(results, f'{results_csv_fn}_{dt}.csv')
    return results_csv

def compute_piece_metric(metric, subset, piece_path, composer, title, split, piece_id = None, audio=None):
    """
    """
    header = ['composer', 'title', 'split',
                    'performer', 'model', 'recording']
    if metric == 'frame':
        header.extend(['p_f', 'r_f', 'f_f'])
    elif metric == 'note_offset':
        header.extend(['p_no', 'r_no', 'f_no'])
    elif metric == 'note_offset_velocity':
        header.extend(['p_nov', 'r_nov', 'f_nov'])
    elif metric == 'dynamics':
        header.append('dyn_corr')
    elif metric == 'articulation':
        header.extend(['melody_kor_corr_64', 'bass_kor_corr_64',  'ratio_kor_corr_64', 'melody_kor_corr_127', 'bass_kor_corr_127', 'ratio_kor_corr_127'])
    elif metric == 'timing':
        header.extend(['melody_ioi_corr', 'acc_ioi_corr', 'ratio_ioi_corr',
                  'melody_ioi_dtw_dist', 'acc_ioi_dtw_dist', 
                  'melody_ioi_hist_kld', 'acc_ioi_hist_kld'])
    elif metric == 'harmony':
        header.extend(['cd_corr', 'cm_corr', 'ts_corr'])        

    results_csv = construct_results_filepath(metric, subset)
    if os.path.exists(results_csv):
        mode = 'a'
    else:
        mode = 'w+'

    if subset != 'revnoise':
        
        files = sorted(os.listdir(piece_path))
        performers = [f.split('.')[0] for f in files if f.endswith('.mid')]
        
        with open(results_csv, mode) as f:
            csvwriter = csv.writer(f)
            if mode == 'w+':
                csvwriter.writerow(header)

            for pidx, performer in enumerate(performers):
                gt_midi = os.path.join(piece_path, f'{performer}.mid')
                gt_perf = pt.load_performance_midi(gt_midi)
                gt_na = gt_perf.note_array()
                if metric == 'frame':
                    gt_pr = pt.utils.compute_pianoroll(gt_na, **PERF_PIANO_ROLL_PARAMS)
                elif metric == 'note_offset' or metric == 'note_offset_velocity':
                    gt_notelist = create_note_list(gt_na)

                print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            
                oa_transcriptions = os.path.join(
                    piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(
                    piece_path, f'{performer}_disklavier')
                
                for rec_env, rec_tr in zip(['maestro', 'disklavier'], 
                    [oa_transcriptions, disklavier_transcriptions]):
                    print(f' --- {rec_env} audio transcriptions:')
                    for tr_mid in os.listdir(rec_tr):
                        if not tr_mid.endswith('.mid'):
                            continue
                        else:
                            print(f' ------ transcription {tr_mid}')
                            tr_midi = os.path.join(rec_tr, tr_mid)
                            tr_perf = pt.load_performance_midi(tr_midi)
                            tr_na = tr_perf.note_array()
                            model = tr_mid.split('.')[0].split('_')[0]
                            
                            if metric in ['frame', 'note_offset', 'note_offset_velocity']:
                                if metric == 'frame':
                                    tr_pr = pt.utils.compute_pianoroll(tr_na, **PERF_PIANO_ROLL_PARAMS)
                                    p, r, f = compute_transcription_benchmark_framewise(gt_pr, tr_pr)
                                else:
                                    tr_notelist = create_note_list(tr_na)
                                    if metric == 'note_offset':
                                        p, r, f, _ = ir_metrics_notewise(gt_notelist, tr_notelist)
                                    elif metric == 'note_offset_velocity':
                                        p, r, f, _ = ir_metrics_notewise_with_velocity(gt_notelist, tr_notelist)
                                    
                                res = [composer, title, split, performer, model, rec_env, p, r, f]
                                
                            elif metric == 'dynamics':
                                dyn_corr = dynamics_metrics_from_perf(tr_perf, gt_perf)
                                res = [composer, title, split, performer, model, rec_env, dyn_corr]
                                
                            elif metric == 'articulation':
                                art_metrics = articulation_metrics_from_perf(tr_perf, gt_perf)
                                art_metrics_64, art_metrics_127 = art_metrics[0], art_metrics[1]
                                melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, _ = art_metrics_64
                                melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127, _ = art_metrics_127
                                res = [composer, title, split, performer, model, rec_env, melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127]
                                
                            elif metric  == 'timing':
                                timing_metrics = timing_metrics_from_perf(tr_perf, gt_perf)
                                res = [composer, title, split, performer, model, rec_env ,*timing_metrics[0]]   
                                
                            elif metric  == 'harmony':
                                harmony_metrics = harmony_metrics_from_perf(tr_perf, gt_perf)
                                res = [composer, title, split, performer, model, rec_env ,*harmony_metrics[0]]

                            res = [np.round(r, 4) if isinstance(r, float) else r for r in res]
                            csvwriter.writerow(res)
    
    
    else: # revnoise subset
        
        header.extend(['reverb', 'snr'])
        header.remove('performer')

        files = [f for f in sorted(os.listdir(piece_path)) if f.startswith(str(piece_id)) and f.endswith('.mid')] # exclude the GT trancsriptions
        gt_midi = [f for f in files if f.startswith(str(piece_id) + 'xx')][0]
        files.remove(gt_midi) # 45 files : 15 (rev,n) combinations x 3 models 
        
        gt_perf = pt.load_performance_midi(os.path.join(piece_path,gt_midi))
        gt_na = gt_perf.note_array()
        if metric == 'frame':
            gt_pr = pt.utils.compute_pianoroll(gt_na, **PERF_PIANO_ROLL_PARAMS)
        elif metric == 'note_offset' or metric == 'note_offset_velocity':
            gt_notelist = create_note_list(gt_na)
        
        with open(results_csv, mode) as f:
            csvwriter = csv.writer(f)
            if mode == 'w+':
                csvwriter.writerow(header)
            
            for i, tr_mid in tqdm(enumerate(files)):
                print(f' --- {i+1:2d}/{len(files)}')
                tr_midi = os.path.join(piece_path, tr_mid)
                tr_perf = pt.load_performance_midi(tr_midi)
                tr_na = tr_perf.note_array()
                if tr_na.shape[0] == 0:
                    print(f'!!!!! EMPTY {tr_mid}')
                    continue
                
                model = tr_mid.split('.')[0].split('_')[-1]
                reverb = REVERB_IRF[int(tr_mid.split('_')[0][1])]
                snr = SNR_LEVELS[int(tr_mid.split('_')[0][2])]
                
                if metric in ['frame', 'note_offset', 'note_offset_velocity']:
                    if metric == 'frame':
                        tr_pr = pt.utils.compute_pianoroll(
                            tr_na, **PERF_PIANO_ROLL_PARAMS)
                        p, r, f = compute_transcription_benchmark_framewise(
                            gt_pr, tr_pr)
                    else:
                        tr_notelist = create_note_list(tr_na)
                        if metric == 'note_offset':
                            p, r, f, _ = ir_metrics_notewise(
                                gt_notelist, tr_notelist)
                        elif metric == 'note_offset_velocity':
                            p, r, f, _ = ir_metrics_notewise_with_velocity(
                                gt_notelist, tr_notelist)

                    res = [composer, title, split, model, audio, p, r, f]

                elif metric == 'dynamics':
                    dyn_corr = dynamics_metrics_from_perf(tr_perf, gt_perf)
                    res = [composer, title, split, model, audio, dyn_corr]
                    
                elif metric == 'articulation':
                    art_metrics = articulation_metrics_from_perf(tr_perf, gt_perf)
                    art_metrics_64, art_metrics_127 = art_metrics[0], art_metrics[1]
                    melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, _ = art_metrics_64
                    melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127, _ = art_metrics_127
                    res = [composer, title, split, model, audio, melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127]
                    
                elif metric  == 'timing':
                    timing_metrics = timing_metrics_from_perf(tr_perf, gt_perf)
                    res = [composer, title, split, model, audio ,*timing_metrics[0]]   
                    
                elif metric  == 'harmony':
                    harmony_metrics = harmony_metrics_from_perf(tr_perf, gt_perf)
                    res = [composer, title, split, model, audio ,*harmony_metrics[0]]
            
                res = [np.round(r, 4) if isinstance(r, float) else r for r in res]
                res.extend([reverb, snr])
                csvwriter.writerow(res)
        
    return None

####################################
# compute metrics for a subset of the eval data
####################################

def compute_subset_metrics(
    subset, subset_meta_csv, eval_set_path, metric_type="frame"
):
    """
    Compute metrics for a given subset of the eval data
    Args:
        subset [str] -- name of the subset
        subset_meta_csv [str] -- path to the meta csv for the subset
        eval_set_path [str] -- path to the eval data
        metric_type [str] -- type of metric to compute
    """
    # read the meta csv
    if not isinstance(subset_meta_csv, pd.DataFrame):
        subset_meta_csv = pd.read_csv(subset_meta_csv)
    
    if subset == "maestro_subset":
        grouping_var = ["split", "folder"]

        for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
            piece_path = os.path.join(
                eval_set_path,
                piece_data["split"].unique()[0],
                piece_data["folder"].unique()[0],
            )
            composer = piece_data["composer"].unique()[0]
            title = piece_data["title"].unique()[0]
            split = piece_data["split"].unique()[0]

            print(69 * "-")
            print(f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")
            
            compute_piece_metric(metric_type,
                    subset, piece_path, composer, title, split)
            
    elif subset == 'revnoise':
        for _, row in tqdm(subset_meta_csv.iterrows()):
            print(69 * "-")
            print(f'Computing {metric_type} for {row["composer"]}, {row["title"]}')
            piece_id = row['piece_id']
            compute_piece_metric(metric_type, subset,
                                 eval_set_path, row['composer'], row['title'], row['split'], piece_id= piece_id, audio=row['audio_env'])
            print()
            
    return None
