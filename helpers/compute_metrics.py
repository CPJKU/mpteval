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

# helpers
def remove_part_prefix_in_note_ids(score_note_array):
    """
    Remove the part prefix in note arrays created from score.xml to match them to those in a match file
    """
    # check if score note ids are correctly encoded
    if "p" in score_note_array["id"][0]:
        score_note_array["id"] = np.char.add(
            "n", np.char.partition(score_note_array["id"], sep="n")[:, -1]
        )
    return score_note_array


def get_measure_boundaries(subset, piece_path, performer):
    """
    Get measure boundaries and onset times for a given performance and piece

    Arguments:
        piece_path [str] -- path to the piece folder
        performer [str] -- performer according to the asap performance midi

    Returns:
        xml_measures [list] -- list of measure numbers according to the unfolded score xml (falls back to match measure numbers when xml != match measures)
        measure_onsets_sec [np.array] -- measure onset times in seconds
        pna [np.array] -- performance note array

        measure_onsets_sec array is also written into measure_onset_times.csv file in piece_path
    """

    # load score
    if subset == "maestro_subset":
        score_xml = os.path.join(piece_path, "xml_score.musicxml")
    # load ground truth for that performances
    gt_match = os.path.join(piece_path, f"{performer}.match")

    # unfold alignment, score and performance for ground truth
    performance, alignment, score = pt.load_match(gt_match, create_score=True)
    pna = performance.note_array()
    pna = pna[np.argsort(pna["onset_sec"])]  # sort by onset

    # get score note array from match score
    part = score.parts[0]
    sna = part.note_array()
    sna = remove_part_prefix_in_note_ids(sna)
    # get score note array from score xml and unfold
    xml_part = pt.load_score(score_xml).parts[0]
    xml_part = pt.score.unfold_part_alignment(xml_part, alignment)
    xml_sna = xml_part.note_array()
    xml_sna = remove_part_prefix_in_note_ids(xml_sna)

    # compare xml score measures with match score measures
    xml_measures = [m for m in xml_part.iter_all(pt.score.Measure)]
    xml_measure_numbers = [m.number for m in xml_measures]
    m_measures = [m for m in part.iter_all(pt.score.Measure)]
    m_measure_numbers = [m.number for m in m_measures]

    if len(xml_measure_numbers) != len(m_measure_numbers):
        # print(f'match has {len(m_measure_numbers)} measures, score has {len(xml_measure_numbers)} measures')
        xml_measure_numbers = m_measure_numbers  # we take the match measure numbers if score and match differ

    # store measure onset times in a csv file
    onset_times_csv = os.path.join(piece_path, "measure_onset_times.csv")
    if os.path.exists(onset_times_csv):
        onset_times_df = pd.read_csv(onset_times_csv)
    else:
        onset_times_df = pd.DataFrame()
        onset_times_df["xml_m"] = xml_measure_numbers

    if f"{performer}_m" in onset_times_df.columns:
        # print(f'Found measure onset times for {performer}')
        return (
            onset_times_df[f"xml_m"].values,
            onset_times_df[f"{performer}_onset_sec"].values,
            pna,
        )
    else:
        onset_times_df[f"{performer}_m"] = m_measure_numbers

        # get the onset sec of each measure (earliest note in the measure)
        measure_onsets_sec = np.zeros(len(m_measures))
        for midx, m in enumerate(m_measures):
            # get all score notes that have their onset in the current measure
            measure_snotes = sna[
                np.where(
                    (sna["onset_div"] >= m.start.t) & (sna["onset_div"] < m.end.t)
                )[0]
            ]
            measure_snids = measure_snotes["id"]
            if measure_snids.shape[0] == 0:
                # there is no measure because there are no notes in the measure
                # print(f'Found no score note in measure {m.number} -- setting onset to NA')
                measure_onsets_sec[midx] = np.nan
                continue

            # find matched performance notes in the measure
            measure_pnids = [
                n["performance_id"]
                for n in alignment
                if n["label"] == "match" and n["score_id"] in measure_snids
            ]
            if measure_pnids == []:
                # print(f'Found no matched performance note in measure {m.number} to match with score notes {measure_snids} -- setting onset to NA')
                measure_onsets_sec[midx] = np.nan
                continue
            # get earliest onset
            measure_pnotes_mask = np.isin(pna["id"], measure_pnids)
            measure_onset_sec = np.min(pna[measure_pnotes_mask]["onset_sec"])
            measure_onsets_sec[midx] = measure_onset_sec
            # ## track measure onset snotes and pnotes
            # measure_onset_snids = measure_snotes[np.where(measure_snotes['onset_beat'] == np.min(measure_snotes['onset_beat']))]['id']
            # measure_onset_pnid = pna[np.where(
            #     pna['onset_sec'] == measure_onset_sec)[0]]['id']
            # measure_onset_snids_list.append(measure_onset_snids)
            # measure_onset_pnids_list.append(measure_onset_pnid[0])

        onset_times_df[f"{performer}_onset_sec"] = np.round(measure_onsets_sec, 4)
        onset_times_df.to_csv(onset_times_csv, index=False)

        return xml_measure_numbers, measure_onsets_sec, pna


def cut_note_array_into_segments(note_array, segment_boundaries_sec):
    """
    Convert a performance note array into a piano roll (88 x num_timesteps) and note list (list of notes as np.voids with fields (onsets, offsets, pitch, velocity)) representation, and split both according to the onset times in segment_boundaries_sec indicating measure boundaries

    Args:
        note_array [np.ndarray] -- performance note array
        segment_boundaries_sec [np.ndarray] -- measure onset times in seconds

    Returns:
        segment_notelists [list] -- list of segmented note lists
        segment_pianorolls [list] -- list of segmented piano rolls
    """
    # cut gt midi accordingly --> note list and piano roll
    # split the piano roll into segments
    indices = np.searchsorted(
        note_array["onset_sec"], segment_boundaries_sec[1:], side="left"
    )
    segment_pnas = np.split(note_array, indices)
    # create note list for each segment
    segment_notelists = [create_note_list(s) for s in segment_pnas]
    # create the piano roll
    piano_roll = pt.utils.compute_pianoroll(note_array, **PERF_PIANO_ROLL_PARAMS)
    # get the measure onset boundaries in frames (and remove silence at the beginning)
    time_div = PERF_PIANO_ROLL_PARAMS["time_div"]
    measure_onset_frames_list = np.round(
        (segment_boundaries_sec - note_array[0]["onset_sec"]) * time_div
    ).astype(int)
    # cut the piano roll
    segment_pianorolls = []
    for i, boundary in enumerate(measure_onset_frames_list):
        if i == len(measure_onset_frames_list) - 1:
            segment_pianoroll = piano_roll[:, boundary:]
        else:
            segment_pianoroll = piano_roll[
                :, boundary : measure_onset_frames_list[i + 1]
            ]
        segment_pianorolls.append(segment_pianoroll)

    return segment_notelists, segment_pianorolls


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
        performers = [f.split('.')[0] for f in files if f.endswith('.match')]
        
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
