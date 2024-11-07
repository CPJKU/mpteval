import numpy as np
import partitura as pt

from scipy.sparse import csc_matrix, hstack
from mir_eval import transcription as mir_eval_transcription
from mir_eval import transcription_velocity as mir_eval_transcription_velocity

import warnings
warnings.filterwarnings("ignore", module="partitura")

PERF_PIANO_ROLL_PARAMS = {
    "time_unit": "sec",
    "time_div": 100,  # frames per time_unit, i.e. with time_div=100 each frame has 10ms resolution
    "onset_only": False,
    "piano_range": True,  # 88 x num_time_steps
    "time_margin": 0,  # amount of padding before and after piano roll
    "return_idxs": False,
}

ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL = (
    5 if PERF_PIANO_ROLL_PARAMS["time_div"] == 100 else 50
)

def plot_piano_roll(piano_roll, params_dict=PERF_PIANO_ROLL_PARAMS, out_path=None):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, figsize=(8, 4))
    ax.imshow(
        piano_roll.toarray(),
        origin="lower",
        cmap="YlGnBu",  # cmap='gray'
        interpolation="nearest",
        aspect="auto",
    )

    if params_dict:
        _, time_div = params_dict["time_unit"], params_dict["time_div"]
        if time_div == 100:
            ax.set_xlabel(f"Time (frame size = 10ms)")
        if time_div == 1000:
            ax.set_xlabel(f"Time (ms)")

    ax.set_ylabel("Piano key")
    if out_path:
        plt.savefig(out_path)
        print(f"Piano roll saved to {out_path}")
    else:
        plt.show()


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


def create_note_list(note_array, remove_silence=True):

    # for empty note arrays, when we cut predicted midis
    if note_array.shape[0] == 0:
        return np.array([])

    first_onset = note_array["onset_sec"][0]

    if remove_silence:
        # remove silence notes
        note_array_no_silence = note_array.copy()
        note_array_no_silence.dtype.names = note_array.dtype.names
        note_array_no_silence["onset_sec"] = (
            note_array_no_silence["onset_sec"] - first_onset
        )
        note_array = note_array_no_silence.copy()

    # 100 -> 10ms time resolution
    time_div = PERF_PIANO_ROLL_PARAMS["time_div"]
    idxs = np.argsort(note_array["onset_sec"])
    onsets = np.round(note_array["onset_sec"] * time_div).astype(int)
    offsets = np.round(
        (note_array["onset_sec"] + note_array["duration_sec"]) * time_div
    ).astype(int)

    pitch = note_array["pitch"]
    velocity = note_array["velocity"]
    note_list = np.column_stack((onsets, offsets, pitch, velocity))
    note_list = note_list[idxs.argsort()]

    return note_list


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
        zero_negative_durations_idxs = np.where(ref_notelist[:, 1] <= ref_notelist[:, 0])[0]
        ref_notelist = np.delete(ref_notelist, zero_negative_durations_idxs, axis=0)

    if np.any(pred_notelist[:, 1] <= pred_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(
            pred_notelist[:, 1] <= pred_notelist[:, 0]
        )[0]
        pred_notelist = np.delete(pred_notelist, zero_negative_durations_idxs, axis=0)
    
    offset_ratio = None if onset_only else 0.2
    precision, recall, f_score, average_overlap_ratio = (
            mir_eval_transcription.precision_recall_f1_overlap(
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


def ir_metrics_notewise_with_velocity(ref_notelist, pred_notelist, verbose=False):

    if pred_notelist.shape[0] == 0 and ref_notelist.shape[0] != 0:
        return 0, 0, 0, 0
    elif pred_notelist.shape[0] != 0 and ref_notelist.shape[0] == 0:
        return 0, 0, 0, 0
    elif pred_notelist.shape[0] == 0 and ref_notelist.shape[0] == 0:
        # no notes in the prediction and no notes in the ground truth
        return 1, 1, 1, 1

    # check if we have zero or negative duration notes in the ground truth or prediction
    if np.any(ref_notelist[:, 1] <= ref_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(ref_notelist[:, 1] <= ref_notelist[:, 0])[0]
        ref_notelist = np.delete(
            ref_notelist, zero_negative_durations_idxs, axis=0)

    if np.any(pred_notelist[:, 1] <= pred_notelist[:, 0]):
        zero_negative_durations_idxs = np.where(
            pred_notelist[:, 1] <= pred_notelist[:, 0])[0]
        pred_notelist = np.delete(
            pred_notelist, zero_negative_durations_idxs, axis=0)

    precision, recall, f_score, average_overlap_ratio = (
        mir_eval_transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_notelist[:, :2],
            ref_pitches=ref_notelist[:, 2],
            ref_velocities=ref_notelist[:, 3],
            est_intervals=pred_notelist[:, :2],
            est_pitches=pred_notelist[:, 2],
            est_velocities=pred_notelist[:, 3],
            onset_tolerance=ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL,
            pitch_tolerance=0,
            offset_ratio=0.2,
            offset_min_tolerance=ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL,
            # beta : float, optional, how much more importance should be placed on recall than precision (1.0 means recall is as important as precision, 2.0 means recall is twice as important as precision, etc.)
            velocity_tolerance=0.1,
            beta=1.0,
            strict=False,
        )
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
