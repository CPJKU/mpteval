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

from .articulation import skyline_melody_identification
from mpteval.utils import is_monophonic, fast_dynamic_time_warping

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
    include_distance: Union[None, Literal['dtw', 'kld']] = None,
) -> np.ndarray:

    timing_metrics = np.zeros(
        1,
        dtype=[
            ("melody_ioi_corr", float),
            ("bass_ioi_corr", float),
            ("ratio_ioi_corr", float),
        ],
    )

    if isinstance(ref_perf, Performance):
        ref_perf = ref_perf.performedparts[0]
    if isinstance(pred_perf, Performance):
        pred_perf = pred_perf.performedparts[0]

    ref_note_array = ref_perf.note_array()
    pred_note_array = pred_perf.note_array()

    if is_monophonic(ref_note_array) and is_monophonic(pred_note_array):

        warnings.warn("Prediction and reference are monophonic, metrics for non-melody stream fallback to nan")

        ref_melody_ioi_func = get_ioi_stream_func(ref_note_array)
        ref_melody_onsets = np.unique(ref_note_array["onset_sec"])
        ref_melody_ioi = ref_melody_ioi_func(ref_melody_onsets)

        pred_melody_ioi_func = get_ioi_stream_func(pred_note_array)
        pred_melody_ioi = pred_melody_ioi_func(ref_melody_onsets)

        timing_metrics["melody_ioi_corr"] = np.corrcoef(ref_melody_ioi, pred_melody_ioi)[0, 1]
        timing_metrics["bass_ioi_corr"] = np.nan
        timing_metrics["ratio_ioi_corr"] = np.nan

    else:

        def split_streams(note_array):
            upper, lower, middle = skyline_melody_identification(note_array)
            melody = upper
            acc = np.concatenate([lower, middle]) if len(middle) > 0 else lower
            return melody, acc

        ref_melody, ref_bass = split_streams(ref_note_array)
        pred_melody, pred_bass = split_streams(pred_note_array)

        ref_melody_onsets = np.unique(ref_melody["onset_sec"])
        ref_bass_onsets = np.unique(ref_bass["onset_sec"])

        ref_melody_ioi = get_ioi_stream_func(ref_melody)(ref_melody_onsets)
        ref_bass_ioi = get_ioi_stream_func(ref_bass)(ref_bass_onsets)

        pred_melody_ioi = get_ioi_stream_func(pred_melody)(ref_melody_onsets)
        pred_bass_ioi = get_ioi_stream_func(pred_bass)(ref_bass_onsets)

        timing_metrics["melody_ioi_corr"] = np.corrcoef(ref_melody_ioi, pred_melody_ioi)[0, 1]
        timing_metrics["bass_ioi_corr"] = np.corrcoef(ref_bass_ioi, pred_bass_ioi)[0, 1]

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
            ref_bass_pr = pt.utils.music.compute_pianoroll(
                note_info=ref_bass,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            ref_melody_features = ref_melody_pr.toarray().T
            ref_bass_features = ref_bass_pr.toarray().T

            pred_melody_pr = pt.utils.music.compute_pianoroll(
                note_info=pred_melody,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            pred_bass_pr = pt.utils.music.compute_pianoroll(
                note_info=pred_bass,
                time_unit="sec",
                time_div=8,
                return_idxs=False,
                piano_range=True,
                binary=True,
                note_separation=True,
            )
            pred_melody_features = pred_melody_pr.toarray().T
            pred_bass_features = pred_bass_pr.toarray().T

            _, melody_dtw_distance = fast_dynamic_time_warping(
                ref_melody_features,
                pred_melody_features,
                metric="cityblock",
                return_distance=True,
            )
            _, acc_dtw_distance = fast_dynamic_time_warping(
                ref_bass_features,
                pred_bass_features,
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
                ref_bass_hist = np.histogram(ref_bass_ioi, bins=bins, density=True)[0]
                pred_bass_hist = np.histogram(pred_bass_ioi, bins=bins, density=True)[0]

                ref_melody_hist[ref_melody_hist == 0] = 1e-6
                pred_melody_hist[pred_melody_hist == 0] = 1e-6
                ref_bass_hist[ref_bass_hist == 0] = 1e-6
                pred_bass_hist[pred_bass_hist == 0] = 1e-6

                # compute the symmetric KLD between the two
                melody_kld = 0.5 * (
                    entropy(ref_melody_hist, pred_melody_hist)
                    + entropy(pred_melody_hist, ref_melody_hist)
                )
                acc_kld = 0.5 * (
                    entropy(ref_bass_hist, pred_bass_hist) + entropy(pred_bass_hist, ref_bass_hist)
                )

                timing_metrics["melody_ioi_hist_kld"] = melody_kld
                timing_metrics["acc_ioi_hist_kld"] = acc_kld

    return timing_metrics
