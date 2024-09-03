"""
Harmony metrics for transcription:

- cloud_diameter_corr : cloud diameter = max dispersion between notes in a window
- cloud_momentum_corr : movement in tonality
- tensile_strain_corr : distances between local and global tonal context

"""

import os
from typing import Callable
import numpy as np

import partitura as pt
from partitura.utils.generic import interp1d
from partitura.performance import PerformedPart
from partitura.musicanalysis.tonal_tension import estimate_tonaltension

import warnings
warnings.filterwarnings('ignore')


def get_tonal_tension_feature_func(note_array, feature, ws, ss, return_onsets=False):

    tonal_tension = estimate_tonaltension(note_array, ws, ss)

    feat_func = interp1d(
        x=tonal_tension['onset_sec'],
        y=tonal_tension['cloud_diameter'],
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=-1,
    )

    if return_onsets:
        return tonal_tension['onset_sec'], tonal_tension[feature], feat_func
    else:
        return tonal_tension[feature], feat_func

def harmony_metrics_from_perf(
    gt_perf: PerformedPart,
    pred_perf: PerformedPart,
    ws = 5, 
    ss = 1, 
) -> np.ndarray:
    """
    Compute the correlation between the tonal tension of the ground truth and predicted performances.
    
    Specifically, we compute the correlation between the following measures of tonal tension:
    - cloud diameter
    - cloud momentum
    - tensile strain
    """

    harmony_metrics = np.zeros(1,
                              dtype=[
                                  ("cloud_diameter_corr", float),
                                  ("cloud_momentum_corr", float),
                                  ("tensile_strain_corr", float),
                              ],
                              )

   
    # get melody and accompaniment for gt performance
    gt_note_array = gt_perf.note_array()
    gt_onsets = np.unique(gt_note_array['onset_sec'])
    
    # get tonal tension func and interpolation function
    gt_onsets, gt_cd, cd_func = get_tonal_tension_feature_func(
        gt_note_array, 'cloud_diameter', ws, ss, return_onsets=True)
    gt_cm, cm_func = get_tonal_tension_feature_func(
        gt_note_array, 'cloud_momentum', ws, ss)
    gt_ts, ts_func = get_tonal_tension_feature_func(
        gt_note_array, 'tensile_strain', ws, ss)
    
    # get tonal tension and interp func for prediction
    tr_note_array = pred_perf.note_array()
    
    tr_cd, tr_func = get_tonal_tension_feature_func(
                tr_note_array, 'cloud_diameter', ws, ss)
    tr_cm, tr_func = get_tonal_tension_feature_func(
        tr_note_array, 'cloud_momentum', ws, ss)
    tr_ts, tr_func = get_tonal_tension_feature_func(
        tr_note_array, 'tensile_strain', ws, ss)
    tr_cd = tr_func(gt_onsets)
    tr_cm = tr_func(gt_onsets)
    tr_ts = tr_func(gt_onsets)
    
    # compute correlation
    cd_corr = np.corrcoef(gt_cd, tr_cd)[0, 1]
    cm_corr = np.corrcoef(gt_cm, tr_cm)[0, 1]
    ts_corr = np.corrcoef(gt_ts, tr_ts)[0, 1]
            
    harmony_metrics['cloud_diameter_corr'] = cd_corr
    harmony_metrics['cloud_momentum_corr'] = cm_corr
    harmony_metrics['tensile_strain_corr'] = ts_corr
    
    return harmony_metrics


# gt_midi = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset/test/Beethoven/Piano_Sonatas/18-1/ChenGuang03M.mid'
# # tr_midi = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset/test/Beethoven/Piano_Sonatas/18-1/ChenGuang03M_disklavier/kong_ChenGuang03M_disklavier.mid'
# tr_midis = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset/test/Beethoven/Piano_Sonatas/18-1/ChenGuang03M_disklavier'


# gt_perf = pt.load_performance_midi(gt_midi)
# gt_note_array = gt_perf.note_array()
# gt_onsets = np.unique(gt_note_array['onset_sec'])

# # gt_cd, cd_func = get_tonal_tension_feature_func(gt_note_array, 'cloud_diameter')
# # gt_cm, cm_func = get_tonal_tension_feature_func(gt_note_array, 'cloud_momentum')
# # gt_ts, ts_func = get_tonal_tension_feature_func(gt_note_array, 'tensile_strain')


# wss = [1, 2, 3, 4, 5]
# sss = [1, 2, 3, 4, 5]
# for ws in wss:
#     for ss in sss:

#         gt_onsets = np.unique(gt_note_array['onset_sec'])

#         gt_onsets, gt_cd, cd_func = get_tonal_tension_feature_func(
#             gt_note_array, 'cloud_diameter', ws, ss, return_onsets=True)
#         gt_cm, cm_func = get_tonal_tension_feature_func(
#             gt_note_array, 'cloud_momentum', ws, ss)
#         gt_ts, ts_func = get_tonal_tension_feature_func(
#             gt_note_array, 'tensile_strain', ws, ss)

#         # print('gt:')
#         print(
#             f'{gt_onsets.shape[0]:4d} onsets -- {gt_cd.shape[0]:3d} tension feats -- ws={ws} ss={ss}')

#         for tr_midi in os.listdir(tr_midis):
#             if tr_midi.endswith('.wav'):
#                 continue

#             model = tr_midi.split('_')[0]

#             # print(tr_midi)
#             tr_perf = pt.load_performance_midi(os.path.join(tr_midis, tr_midi))
#             tr_note_array = tr_perf.note_array()
#             tr_onsets = np.unique(tr_note_array['onset_sec'])
#             # ('onset_sec', 'cloud_diameter', 'cloud_momentum', 'tensile_strain')
#             tr_cd, tr_func = get_tonal_tension_feature_func(
#                 tr_note_array, 'cloud_diameter', ws, ss)
#             tr_cm, tr_func = get_tonal_tension_feature_func(
#                 tr_note_array, 'cloud_momentum', ws, ss)
#             tr_ts, tr_func = get_tonal_tension_feature_func(
#                 tr_note_array, 'tensile_strain', ws, ss)
#             # print(f'{tr_onsets.shape[0]:4d} onsets -- {tr_cd.shape[0]:3d} tension feats')

#             tr_cd = tr_func(gt_onsets)
#             tr_cm = tr_func(gt_onsets)
#             tr_ts = tr_func(gt_onsets)
#             # tr_tonal_tension = estimate_tonaltension(tr_note_array, ws=ws, ss=ss)

#             cd_corr = np.corrcoef(gt_cd, tr_cd)[0, 1]
#             cm_corr = np.corrcoef(gt_cm, tr_cm)[0, 1]
#             ts_corr = np.corrcoef(gt_ts, tr_ts)[0, 1]
#             print(
#                 f'{model:4s} : CD corr: {cd_corr:.3f} -- CM corr: {cm_corr:.3f} -- TS corr: {ts_corr:.3f}')

#     print()

# print()
