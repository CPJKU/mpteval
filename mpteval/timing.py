"""
Timing metrics for transcription:
- IOI correlation for melody and accompaniment streams
- DTW distance between IOI sequences for melody and accompaniment streams
- Kullback-Leibler divergence between IOI histograms for melody and accompaniment streams
"""

from typing import Callable, Union, Literal
import numpy as np
from scipy.stats import entropy
import warnings

import partitura as pt
from partitura.utils.generic import interp1d
from partitura.performance import PerformedPart, Performance

from .articulation import skyline_melody_identification_from_array
from .utils import is_monophonic, fast_dynamic_time_warping


def compute_ioi_stream(note_array: np.ndarray) -> np.ndarray:

    onsets = note_array["onset_sec"]
    sort_idxs = note_array["onset_sec"].argsort()
    ioi = np.zeros(onsets.shape)
    ioi[:-1] = onsets[sort_idxs[1:]] - onsets[sort_idxs[:-1]] + 1e-6
    # add last note duration as last IOI
    ioi[-1] = note_array[sort_idxs[-1]]["duration_sec"] + 1e-6

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


def timing_metrics_from_perf(
    ref_perf: Union[PerformedPart, Performance],
    pred_perf: Union[PerformedPart, Performance],
    include_distance: Union[None, Literal['dtw', 'kld']] =None
) -> np.ndarray:
    
    timing_metrics = np.zeros(
        1,
        dtype=[
            ("melody_ioi_corr", float),
            ("acc_ioi_corr", float),
            ("ratio_ioi_corr", float),
        ],
    )
    
    # check correct types
    if isinstance(ref_perf, Performance):
        ref_perf = ref_perf.performedparts[0]
    if isinstance(pred_perf, Performance):
        pred_perf = pred_perf.performedparts[0]

    ref_note_array = ref_perf.note_array()
    pred_note_array = pred_perf.note_array()
    
    if (is_monophonic(ref_note_array) and not is_monophonic(pred_note_array)) or (not is_monophonic(ref_note_array) and is_monophonic(pred_note_array)):
        # add an alignment step to make reference/prediction monotonic
        raise NotImplementedError()
    
    elif is_monophonic(ref_note_array) and is_monophonic(pred_note_array):
        
        warnings.warn("Prediction and reference are monophonic, metrics for non-melody stream fallback to nan")
        
        ref_melody_onsets = np.unique(ref_note_array["onset_sec"])
        ref_melody_ioi_func = get_ioi_stream_func(ref_note_array)
        ref_melody_ioi = ref_melody_ioi_func(ref_melody_onsets)
        
        pred_melody_onsets = np.unique(pred_note_array["onset_sec"])
        pred_melody_ioi_func = get_ioi_stream_func(pred_note_array)    
        pred_melody_ioi = pred_melody_ioi_func(ref_melody_onsets)
        
        corr_melody_ioi = np.corrcoef(ref_melody_ioi, pred_melody_ioi)[0, 1]
        
        timing_metrics["melody_ioi_corr"] = corr_melody_ioi
        timing_metrics["acc_ioi_corr"] = np.nan
        timing_metrics["ratio_ioi_corr"] = np.nan
        
    else:
        
        # Get melody and accompaniment IOIs for reference performance
        ref_melody, ref_acc = skyline_melody_identification_from_array(ref_note_array)
        ref_melody_onsets = np.unique(ref_melody["onset_sec"])
        ref_acc_onsets = np.unique(ref_acc["onset_sec"])
        
        ref_melody_ioi_func = get_ioi_stream_func(ref_melody)
        ref_acc_ioi_func = get_ioi_stream_func(ref_acc)

        ref_melody_ioi = ref_melody_ioi_func(ref_melody_onsets)
        ref_acc_ioi = ref_acc_ioi_func(ref_acc_onsets)

        # Get melody and accompaniment IOIs for predicted performance
        pred_melody, pred_acc = skyline_melody_identification_from_array(pred_note_array)
        
        pred_melody_ioi_func = get_ioi_stream_func(pred_melody)
        pred_acc_ioi_func = get_ioi_stream_func(pred_acc)
        
        pred_melody_ioi = pred_melody_ioi_func(ref_melody_onsets)
        pred_acc_ioi = pred_acc_ioi_func(ref_acc_onsets)

        # calculate correlation between IOIs
        corr_melody_ioi = np.corrcoef(ref_melody_ioi, pred_melody_ioi)[0, 1]
        corr_accompaniment_ioi = np.corrcoef(ref_acc_ioi, pred_acc_ioi)[0, 1]

        timing_metrics["melody_ioi_corr"] = corr_melody_ioi
        timing_metrics["acc_ioi_corr"] = corr_accompaniment_ioi

        if include_distance == 'dtw':

            # create piano rolls for gt and pred melody and accompaniment note arrays
            ref_melody_pr = pt.utils.music.compute_pianoroll(
                note_info=ref_melody,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            ref_acc_pr = pt.utils.music.compute_pianoroll(
                note_info=ref_acc,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            ref_melody_features = ref_melody_pr.toarray().T
            ref_acc_features = ref_acc_pr.toarray().T

            pred_melody_pr = pt.utils.music.compute_pianoroll(
                note_info=pred_melody,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            pred_acc_pr = pt.utils.music.compute_pianoroll(
                note_info=pred_acc,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            pred_melody_features = pred_melody_pr.toarray().T
            pred_acc_features = pred_acc_pr.toarray().T

            _, melody_dtw_distance = fast_dynamic_time_warping(
                ref_melody_features,
                pred_melody_features,
                metric="cityblock",
                return_distance=True,
            )
            _, acc_dtw_distance = fast_dynamic_time_warping(
                ref_acc_features,
                pred_acc_features,
                metric="cityblock",
                return_distance=True,
            )

            timing_metrics["melody_ioi_dtw_dist"] = melody_dtw_distance
            timing_metrics["acc_ioi_dtw_dist"] = acc_dtw_distance

        if include_distance == 'kld':
                # Histogram distance (symmetric KLD)

                # compute histograms for melody and accompaniment IOIs
                # bin size = 10ms for IOIs below 100ms and 100ms from 100ms to 2s
                bins = [i * 0.01 for i in range(10)]
                bins += [0.1 + i * 0.1 for i in range(20)]
                ref_melody_hist = np.histogram(ref_melody_ioi, bins=bins, density=True)[0]
                pred_melody_hist = np.histogram(pred_melody_ioi, bins=bins, density=True)[0]
                ref_acc_hist = np.histogram(ref_acc_ioi, bins=bins, density=True)[0]
                pred_acc_hist = np.histogram(pred_acc_ioi, bins=bins, density=True)[0]

                ref_melody_hist[ref_melody_hist == 0] = 1e-6
                pred_melody_hist[pred_melody_hist == 0] = 1e-6
                ref_acc_hist[ref_acc_hist == 0] = 1e-6
                pred_acc_hist[pred_acc_hist == 0] = 1e-6

                # compute the symmetric KLD between the two
                melody_kld = 0.5 * (
                    entropy(ref_melody_hist, pred_melody_hist)
                    + entropy(pred_melody_hist, ref_melody_hist)
                )
                acc_kld = 0.5 * (
                    entropy(ref_acc_hist, pred_acc_hist) + entropy(pred_acc_hist, ref_acc_hist)
                )

                timing_metrics["melody_ioi_hist_kld"] = melody_kld
                timing_metrics["acc_ioi_hist_kld"] = acc_kld

    return timing_metrics
