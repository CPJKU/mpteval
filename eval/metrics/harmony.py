"""
Harmony metrics for transcription
"""

import numpy as np
from typing import Callable, Tuple, Union

import partitura as pt
from partitura.utils.generic import interp1d
from partitura.performance import PerformedPart, Performance
from partitura.musicanalysis.tonal_tension import estimate_tonaltension

import warnings

warnings.filterwarnings("ignore")


def get_tonal_tension_feature_func(
    note_array: np.ndarray, feature: str, ws: int, ss: int
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Compute a tonal tension feature and return the interpolation function for that feature.

    Parameters
    ----------
    note_array : np.ndarray
        Note array for the stream
    feature : string
        The tonal tension feature to compute. One of 'cloud_diameter', 'cloud_momentum', 'tensile_strain' (see [1]_).
    ws : int
        Window size for tonal tension estimation
    ss : int
        Step size for tonal tension estimation

    Returns
    -------
    tonal_tension[feature] : structured array
        Array containing the tonal tension feature values.
    feat_func : Callable[[np.ndarray], np.ndarray]

    References
    ----------
    .. [1] D. Herremans and E. Chew (2016) Tension ribbons: Quantifying and
           visualising tonal tension. Proceedings of the Second International
           Conference on Technologies for Music Notation and Representation
           (TENOR), Cambridge, UK.
    """

    tonal_tension = estimate_tonaltension(note_array, ws, ss)

    feat_func = interp1d(
        x=tonal_tension["onset_sec"],
        y=tonal_tension[feature],
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=-1,
    )

    return tonal_tension[feature], feat_func


def harmony_metrics_from_perf(
    ref_perf: Union[PerformedPart, Performance],
    pred_perf: Union[PerformedPart, Performance],
    ws: int = 5,
    ss: int = 1,
) -> np.ndarray:

    harmony_metrics = np.zeros(
        1,
        dtype=[
            ("cloud_diameter_corr", float),
            ("cloud_momentum_corr", float),
            ("tensile_strain_corr", float),
        ],
    )

    # get melody and accompaniment for reference performance
    ref_note_array = ref_perf.note_array()
    ref_onsets = np.unique(ref_note_array["onset_sec"])

    # get tonal tension func and interpolation function
    ref_cd, _ = get_tonal_tension_feature_func(ref_note_array, "cloud_diameter", ws, ss)
    ref_cm, _ = get_tonal_tension_feature_func(ref_note_array, "cloud_momentum", ws, ss)
    ref_ts, _ = get_tonal_tension_feature_func(ref_note_array, "tensile_strain", ws, ss)

    # get tonal tension and interp func for prediction
    pred_note_array = pred_perf.note_array()

    _, pred_cd_func = get_tonal_tension_feature_func(
        pred_note_array, "cloud_diameter", ws, ss
    )
    _, pred_cm_func = get_tonal_tension_feature_func(
        pred_note_array, "cloud_momentum", ws, ss
    )
    _, pred_ts_func = get_tonal_tension_feature_func(
        pred_note_array, "tensile_strain", ws, ss
    )
    pred_cd = pred_cd_func(ref_onsets)
    pred_cm = pred_cm_func(ref_onsets)
    pred_ts = pred_ts_func(ref_onsets)

    # compute correlation
    cd_corr = np.corrcoef(ref_cd, pred_cd)[0, 1]
    cm_corr = np.corrcoef(ref_cm, pred_cm)[0, 1]
    ts_corr = np.corrcoef(ref_ts, pred_ts)[0, 1]

    harmony_metrics["cloud_diameter_corr"] = cd_corr
    harmony_metrics["cloud_momentum_corr"] = cm_corr
    harmony_metrics["tensile_strain_corr"] = ts_corr

    return harmony_metrics
