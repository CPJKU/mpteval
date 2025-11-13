'''
Standard frame and notewise transcription metrics from mir_eval, 
with added onset+velocity (no offset) notewise metric variation option.
'''

import numpy as np
import partitura as pt

from scipy.sparse import csc_matrix, hstack
from mir_eval import transcription as mir_eval_transcription
from mir_eval import transcription_velocity as mir_eval_transcription_velocity

from .utils import ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL
import warnings


def compute_transcription_benchmark_framewise(
    ref_piano_roll, pred_piano_roll, verbose=False
):
    time_diff = ref_piano_roll.shape[1] - pred_piano_roll.shape[1]
    padding_csc = csc_matrix((ref_piano_roll.shape[0], abs(time_diff)), dtype=np.int8)

    if time_diff > 0:
        pred_piano_roll = hstack((pred_piano_roll, padding_csc))
    else:
        ref_piano_roll = hstack((ref_piano_roll, padding_csc))

    ref_piano_roll = ref_piano_roll.astype("bool").toarray()
    pred_piano_roll = pred_piano_roll.astype("bool").toarray()
    true_positives = np.sum(np.logical_and(ref_piano_roll == 1, pred_piano_roll == 1))
    false_positives = np.sum(np.logical_and(ref_piano_roll == 0, pred_piano_roll == 1))
    false_negatives = np.sum(np.logical_and(ref_piano_roll == 1, pred_piano_roll == 0))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 * precision * recall / (precision + recall)

    if verbose:
        print(42 * "-")
        print(f"framewise eval")
        print(42 * "-")
        print(f"      precision: {precision:.4f}")
        print(f"         recall: {recall:.4f}")
        print(f"         fscore: {f_score:.4f}")

    return precision, recall, f_score


def ir_metrics_notewise(ref_notelist, pred_notelist, onset_only=False, verbose=False):

    if pred_notelist.shape[0] == 0 and ref_notelist.shape[0] != 0:
        return 0, 0, 0, 0
    elif pred_notelist.shape[0] != 0 and ref_notelist.shape[0] == 0:
        return 0, 0, 0, 0
    elif pred_notelist.shape[0] == 0 and ref_notelist.shape[0] == 0:
        # no notes in the prediction and no notes in the ground truth
        return 1, 1, 1, 1

    # check if we have zero or negative duration notes in the ground truth or prediction
    if np.any(ref_notelist[:, 1] <= ref_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(
            ref_notelist[:, 1] <= ref_notelist[:, 0]
        )[0]
        ref_notelist = np.delete(ref_notelist, zero_negative_durations_idxs, axis=0)

    if np.any(pred_notelist[:, 1] <= pred_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(
            pred_notelist[:, 1] <= pred_notelist[:, 0]
        )[0]
        pred_notelist = np.delete(pred_notelist, zero_negative_durations_idxs, axis=0)

    offset_ratio = None if onset_only else 0.2
    (
        precision,
        recall,
        f_score,
        average_overlap_ratio,
    ) = mir_eval_transcription.precision_recall_f1_overlap(
        ref_intervals=ref_notelist[:, :2],
        ref_pitches=ref_notelist[:, 2],
        est_intervals=pred_notelist[:, :2],
        est_pitches=pred_notelist[:, 2],
        onset_tolerance=ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL,
        pitch_tolerance=0,
        offset_ratio=offset_ratio,
        offset_min_tolerance=ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL,
        # beta : float, optional, how much more importance should be placed on recall than precision (1.0 means recall is as important as precision, 2.0 means recall is twice as important as precision, etc.)
        beta=1.0,
        strict=False,
    )

    if verbose:
        print(42 * "-")
        print(f"{'notewise eval' if onset_only else 'with offset'}")
        print(42 * "-")
        print(f"      precision: {precision:.4f}")
        print(f"         recall: {recall:.4f}")
        print(f"         fscore: {f_score:.4f}")
        print(f"average overlap: {average_overlap_ratio:.4f}")

    return precision, recall, f_score, average_overlap_ratio


def ir_metrics_notewise_with_velocity(ref_notelist, pred_notelist, onset_only=False, verbose=False):

    if pred_notelist.shape[0] == 0 and ref_notelist.shape[0] != 0:
        return 0, 0, 0, 0
    elif pred_notelist.shape[0] != 0 and ref_notelist.shape[0] == 0:
        return 0, 0, 0, 0
    elif pred_notelist.shape[0] == 0 and ref_notelist.shape[0] == 0:
        # no notes in the prediction and no notes in the ground truth
        return 1, 1, 1, 1

    # check if we have zero or negative duration notes in the ground truth or prediction
    if np.any(ref_notelist[:, 1] <= ref_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(
            ref_notelist[:, 1] <= ref_notelist[:, 0]
        )[0]
        ref_notelist = np.delete(ref_notelist, zero_negative_durations_idxs, axis=0)

    if np.any(pred_notelist[:, 1] <= pred_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(
            pred_notelist[:, 1] <= pred_notelist[:, 0]
        )[0]
        pred_notelist = np.delete(pred_notelist, zero_negative_durations_idxs, axis=0)

    offset_ratio = None if onset_only else 0.2
    (
        precision,
        recall,
        f_score,
        average_overlap_ratio,
    ) = mir_eval_transcription_velocity.precision_recall_f1_overlap(
        ref_intervals=ref_notelist[:, :2],
        ref_pitches=ref_notelist[:, 2],
        ref_velocities=ref_notelist[:, 3],
        est_intervals=pred_notelist[:, :2],
        est_pitches=pred_notelist[:, 2],
        est_velocities=pred_notelist[:, 3],
        onset_tolerance=ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL,
        pitch_tolerance=0,
        offset_ratio=offset_ratio,
        offset_min_tolerance=ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL,
        # beta : float, optional, how much more importance should be placed on recall than precision (1.0 means recall is as important as precision, 2.0 means recall is twice as important as precision, etc.)
        velocity_tolerance=0.1,
        beta=1.0,
        strict=False,
    )

    if verbose:
        print(42 * "-")
        print(f"notewise eval")
        print(42 * "-")
        print(f"      precision: {precision:.4f}")
        print(f"         recall: {recall:.4f}")
        print(f"         fscore: {f_score:.4f}")
        print(f"average overlap: {average_overlap_ratio:.4f}")

    return precision, recall, f_score, average_overlap_ratio
