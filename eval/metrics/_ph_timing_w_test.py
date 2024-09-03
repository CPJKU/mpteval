#%%

"""
Timing metrics for transcription
When computing IOIs we use a small constant (1e-6) to avoid division by zero.
"""

import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
import numpy as np
from typing import Callable

import partitura as pt
from partitura.utils.generic import interp1d
from partitura.performance import PerformedPart

from helpers.dtw import fast_dynamic_time_warping
from eval.metrics.articulation import skyline_melody_identification_from_array


def compute_ioi_stream(note_array):
    
    onsets = note_array['onset_sec']
    sort_idxs = note_array['onset_sec'].argsort()
    ioi = np.zeros(onsets.shape)
    ioi[:-1] = onsets[sort_idxs[1:]] - onsets[sort_idxs[:-1]] + 1e-6
    # add last note duration to last ioi
    ioi[-1] = note_array[sort_idxs[-1]]['duration_sec'] + 1e-6 
    
    return ioi


def get_ioi_stream_func(note_array: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:

    ioi = compute_ioi_stream(note_array)
    
    ioi_stream_func = interp1d(
        x=note_array["onset_sec"],
        y=ioi,
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=-1,
    )

    return ioi_stream_func


def get_ioi_stream_funcs(note_array):

    melody, accompaniment = skyline_melody_identification_from_array(note_array)
    
    ioi_melody_func = get_ioi_stream_func(melody)
    ioi_accompaniment_func = get_ioi_stream_func(accompaniment)
    
    return ioi_melody_func, ioi_accompaniment_func


def timing_metrics_from_perf(
    gt_perf: PerformedPart,
    pred_perf: PerformedPart,
) -> np.ndarray:    
    
    timing_metrics = np.zeros(1,
                              dtype=[
                                  ("melody_ioi_corr", float),
                                  ("acc_ioi_corr", float),
                                  ("ratio_ioi_corr", float),

                                  ("melody_ioi_dtw_dist", float),
                                  ("acc_ioi_dtw_dist", float),

                                  ("melody_ioi_hist_kld", float),
                                  ("acc_ioi_hist_kld", float),
                              ],
                              )
    

    ############################
    # compute IOI sequence
    ############################
    # get melody and accompaniment for gt performance
    gt_note_array = gt_perf.note_array()
    melody, accompaniment = skyline_melody_identification_from_array(gt_note_array)
    # get melody and accompaniment onsets for interpolation
    melody_onsets = np.unique(melody['onset_sec'])
    accompaniment_onsets = np.unique(accompaniment['onset_sec'])
    
    # get interpolation function for melody and accompaniment
    ioi_melody_func = get_ioi_stream_func(melody)
    ioi_accompaniment_func = get_ioi_stream_func(accompaniment)
    
    # get interpolated IOIs for melody and accompaniment
    melody_ioi_gt = ioi_melody_func(melody_onsets)
    accompaniment_ioi_gt = ioi_accompaniment_func(accompaniment_onsets)
    
    # get melody and accompaniment from prediction
    pred_note_array = pred_perf.note_array()
    melody_pred, accompaniment_pred = skyline_melody_identification_from_array(pred_note_array)
    # get interpolation function for predicted melody and accompaniment
    ioi_melody_pred_func = get_ioi_stream_func(melody_pred)
    ioi_accompaniment_pred_func = get_ioi_stream_func(accompaniment_pred)
    # get interpolated IOIs for predicted melody and accompaniment
    melody_ioi_pred = ioi_melody_pred_func(melody_onsets)
    accompaniment_ioi_pred = ioi_accompaniment_pred_func(accompaniment_onsets)
    
    ############################
    # Correlations
    ############################
    # calculate correlation between IOIs
    corr_melody_ioi = np.corrcoef(melody_ioi_gt, melody_ioi_pred)[0, 1]
    corr_accompaniment_ioi = np.corrcoef(accompaniment_ioi_gt, accompaniment_ioi_pred)[0, 1]
    
    timing_metrics['melody_ioi_corr'] = corr_melody_ioi
    timing_metrics['acc_ioi_corr'] = corr_accompaniment_ioi
    
    ############################
    # DTW distances
    ############################
    
    # create piano rolls for gt and pred melody and accompaniment note arrays
    melody_gt_pr = pt.utils.music.compute_pianoroll(
        note_info=melody,
        time_unit="sec",
        time_div=8,
        return_idxs=False,
        piano_range=True,
        binary=True,
        note_separation=True,
    )
    accompaniment_gt_pr = pt.utils.music.compute_pianoroll(
        note_info=accompaniment,
        time_unit="sec",
        time_div=8,
        return_idxs=False,
        piano_range=True,
        binary=True,
        note_separation=True,
    )
    melody_gt_features = melody_gt_pr.toarray().T
    accompaniment_gt_features = accompaniment_gt_pr.toarray().T
    
    melody_pred_pr = pt.utils.music.compute_pianoroll(
        note_info=melody_pred,
        time_unit="sec",
        time_div=8,
        return_idxs=False,
        piano_range=True,
        binary=True,
        note_separation=True,
    )
    accompaniment_pred_pr = pt.utils.music.compute_pianoroll(
        note_info=accompaniment_pred,
        time_unit="sec",
        time_div=8,
        return_idxs=False,
        piano_range=True,
        binary=True,
        note_separation=True,
    )
    melody_pred_features = melody_pred_pr.toarray().T
    accompaniment_pred_features = accompaniment_pred_pr.toarray().T
    
    _, dtw_melody_distance = fast_dynamic_time_warping(
        melody_gt_features, melody_pred_features, metric="cityblock", return_distance=True)
    _, dtw_acc_distance = fast_dynamic_time_warping(
        accompaniment_gt_features, accompaniment_pred_features, metric="cityblock", return_distance=True)
    
    timing_metrics['melody_ioi_dtw_dist'] = dtw_melody_distance
    timing_metrics['acc_ioi_dtw_dist'] = dtw_acc_distance
    
    ############################
    # Histogram distance (symmetric KLD)
    ############################
    
    # compute histograms for melody and accompaniment IOIs

    # generate bins
    bins = [i*0.01 for i in range(10)]
    bins += [0.1+i*0.1 for i in range(20)]
    melody_hist_gt = np.histogram(melody_ioi_gt, bins=bins, density=True)[0]
    melody_hist_pred = np.histogram(melody_ioi_pred, bins=bins, density=True)[0]
    acc_hist_gt = np.histogram(accompaniment_ioi_gt, bins=bins, density=True)[0]
    acc_hist_pred = np.histogram(accompaniment_ioi_pred, bins=bins, density=True)[0]
    
    melody_hist_gt[melody_hist_gt == 0] = 1e-6
    melody_hist_pred[melody_hist_pred == 0] = 1e-6
    acc_hist_gt[acc_hist_gt == 0] = 1e-6
    acc_hist_pred[acc_hist_pred == 0] = 1e-6
        
    # compute the symmetric KLD between the two
    melody_kld = 0.5 * (entropy(melody_hist_gt, melody_hist_pred) +
                        entropy(melody_hist_pred, melody_hist_gt))
    acc_kld = 0.5 * (entropy(acc_hist_gt, acc_hist_pred) +
                     entropy(acc_hist_pred, acc_hist_gt))
    
    timing_metrics['melody_ioi_hist_kld'] = melody_kld
    timing_metrics['acc_ioi_hist_kld'] = acc_kld
    
    return timing_metrics


#%%
# gt_test_mid = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset/train/Bach/Fugue/bwv_848/Denisova06M.mid'
# maestro_tr_mid_dir = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset/train/Bach/Fugue/bwv_848/Denisova06M_maestro'
# disklavier_tr_mid_dir = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset/train/Bach/Fugue/bwv_848/Denisova06M_disklavier'

# maestro_transcriptions = [f for f in sorted(os.listdir(maestro_tr_mid_dir)) if f.endswith('.mid')]
# disklavier_transcriptions = [f for f in sorted(os.listdir(disklavier_tr_mid_dir)) if f.endswith('.mid')]
# T5_maestro_mid = os.path.join(maestro_tr_mid_dir, maestro_transcriptions[0])


# gt_perf = pt.load_performance_midi(gt_test_mid)
# tr_perf = pt.load_performance_midi(T5_maestro_mid)

# # get melody and accompaniment for gt performance
# gt_note_array = gt_perf.note_array()
# melody, accompaniment = skyline_melody_identification_from_array(gt_note_array)
# # get melody and accompaniment onsets for interpolation
# melody_onsets = np.unique(melody['onset_sec'])
# accompaniment_onsets = np.unique(accompaniment['onset_sec'])

# # get interpolation function for melody and accompaniment
# ioi_melody_func = get_ioi_stream_func(melody)
# ioi_accompaniment_func = get_ioi_stream_func(accompaniment)

# # get interpolated IOIs for melody and accompaniment
# melody_ioi_gt = ioi_melody_func(melody_onsets)
# accompaniment_ioi_gt = ioi_accompaniment_func(accompaniment_onsets)

# # get melody and accompaniment from prediction
# pred_note_array = tr_perf.note_array()
# melody_pred, accompaniment_pred = skyline_melody_identification_from_array(pred_note_array)
# # get interpolation function for predicted melody and accompaniment
# ioi_melody_pred_func = get_ioi_stream_func(melody_pred)
# ioi_accompaniment_pred_func = get_ioi_stream_func(accompaniment_pred)
# # get interpolated IOIs for predicted melody and accompaniment
# melody_ioi_pred = ioi_melody_pred_func(melody_onsets)
# accompaniment_ioi_pred = ioi_accompaniment_pred_func(accompaniment_onsets)
    
# # generate bins
# bins = [i*0.01 for i in range(10)]
# bins += [0.1+i*0.1 for i in range(20)]
# melody_hist_gt = np.histogram(melody_ioi_gt, bins=bins, density=True)[0]
# melody_hist_pred = np.histogram(melody_ioi_pred, bins=bins, density=True)[0]
# acc_hist_gt = np.histogram(accompaniment_ioi_gt, bins=bins, density=True)[0]
# acc_hist_pred = np.histogram(accompaniment_ioi_pred, bins=bins, density=True)[0]

# melody_hist_gt[melody_hist_gt == 0] = 1e-6
# melody_hist_pred[melody_hist_pred == 0] = 1e-6
# acc_hist_gt[acc_hist_gt == 0] = 1e-6
# acc_hist_pred[acc_hist_pred == 0] = 1e-6

# # compute the symmetric KLD between the two
# melody_kld = 0.5 * (entropy(melody_hist_gt, melody_hist_pred) +
#                 entropy(melody_hist_pred, melody_hist_gt))
# acc_kld = 0.5 * (entropy(acc_hist_gt, acc_hist_pred) +
#                 entropy(acc_hist_pred, acc_hist_gt))

# _ = plt.hist(melody_ioi_gt, bins=bins)  # arguments are passed to np.histogram
# plt.show()
# _ = plt.hist(melody_ioi_pred, bins=bins)  # arguments are passed to np.histogram
# plt.show()
# #%%
# _ = plt.hist(accompaniment_ioi_gt, bins=bins)  # arguments are passed to np.histogram
# plt.show()
# #%%
# _ = plt.hist(accompaniment_ioi_pred, bins=bins)  # arguments are passed to np.histogram
# plt.show()





# #%%
# # timing_metrics_from_perf(tr_perf, gt_perf)


# # '''
# # [NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE][NOTE]

# # [MELODY]
# # - We use the skyline melody identification algorithm to estimate the melody line from the transcription and the ground truth MIDI files.

# # [ARTICULATION] --> Key Overlap Ratio
# # - Melody KOR
# # - Bass KOR
# # - Ratio KOR (melody kor / bass kor) 

# # [TIMING / RHYTHMIC PRECISION]
# # - IOI corr for melody and accompaniment      
# # - Histogram sym KLDs for melody and accompaniment
# # - DTW distance for melody / bass / melody/bass ratio            TODO 




# # '''
