"""
Articulation-based metrics for transcription
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Callable

from partitura.utils.generic import interp1d
from partitura.utils.music import get_time_units_from_note_array
from partitura.performance import Performance, PerformedPart

from .dynamics import PerformedChord
from .util import is_monophonic

import warnings

def chordify_perf_note_array(
    note_array: np.ndarray,
    ioi_threshold: float = 0.03,
    max_threshold: float = 0.05,
) -> List[PerformedChord]:
    """
    Chordify a performance note array.

    Parameters
    ----------
    note_array : np.ndarray
        An input note array
    ioi_threshold : float, optional
        Maximal Inter-onset interval between notes in the chord,
        in seconds, by default 0.03
    max_threshold : float, optional
        Maximal value between the onset time of the first
        and last note in the chord, by default 0.05

    Returns
    -------
    chords : List[PerformedChord]
        List of performed chords.
    """
    sort_idx = note_array["onset_sec"].argsort()

    note_array = note_array[sort_idx]

    chords = [
        PerformedChord(
            pitch=[note_array[0]["pitch"]],
            ponsets=[note_array[0]["onset_sec"]],
            pduration=[note_array[0]["duration_sec"]],
            velocities=[note_array[0]["velocity"]],
            ids=[note_array[0]["id"]],
            chord_id="c0",
            max_threshold=max_threshold,
            ioi_threshold=ioi_threshold,
        )
    ]

    cid = 1
    for note in note_array[1:]:

        added_note = chords[-1].add(
            pitch=note["pitch"],
            onset=note["onset_sec"],
            duration=note["duration_sec"],
            velocity=note["velocity"],
            nid=note["id"],
        )

        if not added_note:

            chord = PerformedChord(
                pitch=[note["pitch"]],
                ponsets=[note["onset_sec"]],
                pduration=[note["duration_sec"]],
                velocities=[note["velocity"]],
                ids=[note["id"]],
                chord_id=f"c{cid}",
                max_threshold=max_threshold,
                ioi_threshold=ioi_threshold,
            )
            chords.append(chord)
            cid += 1

    return chords


def skyline_melody_identification_from_array(
    note_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main melody line identification using the skyline algorithm.

    The skyline algorithm was first proposed by Uitdenbogerd and Zobel [1]_
    and has seen various implementations. The implementation referenced
    here is adapted from Simonetta et al. [2]_.

    Parameters
    ----------
    note_array : np.ndarray
        Input note array (from a PerformedPart or Part).

    Returns
    -------
    melody_line : np.ndarray
        A note array corresponding to the notes in the melody.

    accompaniment : np.ndarray:
        A note array corresponding to the notes in the accompaniment
        (i.e., everything that is not in the main melody line)

    References
      ----------
      .. [1] A. L. Uitdenbogerd and J. Zobel, "Melodic matching techniques for
         large music databases," in Proceedings of the 7th ACM International
         Conference on Multimedia '99, Orlando, FL, USA, October 30 - November 5,
         1999, Part 1, pp. 57-66.

      .. [2] F. Simonetta, C. Cancino-Chacón, S. Ntalampiras and G. Widmer "A Convolutional
         Approach to Melody Line Identification in Symbolic Scores," in Proceedings of the
         20th International Society for Music Information Retrieval Conference (ISMIR 2019),
         Delft, The Netherlands.
    """

    onset_unit, duration_sec = get_time_units_from_note_array(note_array)

    note_info = np.column_stack(
        [
            note_array["pitch"],
            note_array[onset_unit],
            note_array[onset_unit] + note_array[duration_sec],
        ]
    )

    # sort by onset
    sort_idx = note_info[:, 1].argsort()
    back_sort_idxs = sort_idx.argsort()
    note_info = note_info[sort_idx]

    melody_mask = np.zeros(len(note_info), dtype=bool)
    previous_onset = np.inf  # the first time is not a new onset
    highest_pitch = -1
    melody_index = 0
    last_melody_pitch = 0
    for i, (pitch, onset, _) in enumerate(note_info):
        if pitch > highest_pitch:
            # look for the highest pitch among notes at this offset
            highest_pitch = pitch
            melody_index = i
        elif onset > previous_onset:
            # this is a new onset:
            # test if among notes at the previous onset there is a melody note
            if (
                highest_pitch > last_melody_pitch
                or previous_onset >= last_melody_offset
            ):
                # mark the new melody note
                melody_mask[melody_index] = True
                last_melody_offset = note_info[melody_index][2]
                last_melody_pitch = note_info[melody_index][0]
            highest_pitch = 0
        previous_onset = onset

    melody_line = note_array[melody_mask[back_sort_idxs]]

    accompaniment = note_array[~melody_mask[back_sort_idxs]]
    return melody_line, accompaniment


def skyline_melody_identification(
    note_array: np.ndarray,
    staff_threshold=60,
    ioi_threshold: float = 0.03,
    max_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    chords = chordify_perf_note_array(
        note_array=note_array,
        ioi_threshold=ioi_threshold,
        max_threshold=max_threshold,
    )

    upper_voice = []
    lower_voice = []
    middle_voices = []

    for chord in chords:

        max_pitch_idx = np.argmax(chord.pitch)
        min_pitch_idx = np.argmin(chord.pitch)
        upper_mask = note_array["id"] == chord.ids[max_pitch_idx]
        lower_mask = note_array["id"] == chord.ids[min_pitch_idx]
        middle_ids = [
            nid
            for nid in chord.ids
            if nid != chord.ids[max_pitch_idx] and nid != chord.ids[min_pitch_idx]
        ]

        middle_idxs = np.array(
            [int(np.where(note_array["id"] == nid)[0][0]) for nid in middle_ids], dtype=int
        )
        
        max_note = note_array[upper_mask]
        min_note = note_array[lower_mask]
        middle_notes = note_array[middle_idxs]

        if chord.pitch[max_pitch_idx] > staff_threshold:
            upper_voice.append(max_note)

        if max_pitch_idx != min_pitch_idx:

            lower_voice.append(min_note)

        if len(middle_notes) > 0:
            middle_voices.append(middle_notes)

    empty_voice = np.array([], dtype=note_array.dtype)

    upper_voice = np.hstack(upper_voice).flatten() if upper_voice else empty_voice

    lower_voice = np.hstack(lower_voice).flatten() if lower_voice else empty_voice

    middle_voices = np.hstack(middle_voices).flatten() if middle_voices else empty_voice

    return upper_voice, lower_voice, middle_voices


def compute_kor_stream(
    note_array: np.ndarray,
    clip: Optional[float] = 100.0,
) -> np.ndarray:
    """
    Compute key overlap ratio for a mostly monophonic stream

    Parameters
    ----------
    note_array : np.ndarray
        Note array for the stream
    clip : Optional[float], optional
        For normalization purposes we empirically cap the maximum to this value, by default 10.0

    Returns
    -------
    kor: np.ndarray
        Key overlap ratio for each of the notes in the note_array.
    """
    onset_unit, duration_unit = get_time_units_from_note_array(note_array)

    onsets = note_array[onset_unit]
    offsets = note_array[onset_unit] + note_array[duration_unit]

    sort_idxs = note_array[onset_unit].argsort()
    reversed_sort_idxs = sort_idxs.argsort()

    kot = offsets[sort_idxs[1:]] - onsets[sort_idxs[:-1]]
    ioi = onsets[sort_idxs[1:]] - onsets[sort_idxs[:-1]] + 1e-6

    kor = kot / ioi

    if clip is not None:
        np.clip(kor, a_min=None, a_max=clip, out=kor)

    kor = np.r_[kor[reversed_sort_idxs[:-1]], -1]

    return kor


def get_kor_stream_func(note_array: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:

    kor = compute_kor_stream(note_array)

    kor_stream_func = interp1d(
        x=note_array["onset_sec"],
        y=kor,
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=-1,
    )

    return kor_stream_func


def get_kor_stream_funcs(
    note_array: np.ndarray,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:

    melody_line, bass_line, _ = skyline_melody_identification(note_array=note_array)

    kor_melody_func = get_kor_stream_func(melody_line)
    kor_bass_func = get_kor_stream_func(bass_line)

    return kor_melody_func, kor_bass_func


def articulation_metrics_from_perf(
    ref_perf: Union[PerformedPart, Performance],
    pred_perf: Union[PerformedPart, Performance],
    pedal_range: List[int] = [64, 127],
) -> np.ndarray:

    # check correct types
    if isinstance(ref_perf, Performance):
        ref_pparts = ref_perf.performedparts
    elif isinstance(ref_perf, PerformedPart):
        ref_pparts = [ref_perf]

    if isinstance(pred_perf, Performance):
        pred_pparts = pred_perf.performedparts
    elif isinstance(pred_perf, PerformedPart):
        pred_pparts = [pred_perf]

    # adjust sustain pedal threshold
    for ppart in ref_pparts:
        ppart.sustain_pedal_threshold = 64

    # get reference note array and reference KOR functions for melody and bass stream
    ref_note_array = ref_perf.note_array()
    ref_melody_kor_func, ref_bass_kor_func = get_kor_stream_funcs(ref_note_array)

    ref_onsets = np.unique(ref_note_array["onset_sec"])

    # get reference KOR values for melody and bass stream, and their ratio
    ref_melody_kor = ref_melody_kor_func(ref_onsets)
    ref_bass_kor = ref_bass_kor_func(ref_onsets)
    ref_ratio_kor = ref_melody_kor / ref_bass_kor

    # init and compute metrics array
    metrics = np.zeros(
        len(pedal_range),
        dtype=[
            ("melody_kor_corr", float),
            ("bass_kor_corr", float),
            ("ratio_kor_corr", float),
            ("pedal_threshold", int),
        ],
    )
    for i, spt in enumerate(pedal_range):
        
        metrics[i]["pedal_threshold"] = spt

        # adjust sustain pedal threshold
        for ppart in pred_pparts:
            ppart.sustain_pedal_threshold = spt

        # get predicted note array and KOR functions, and values for melody and bass stream
        pred_note_array = pred_perf.note_array()
        
        if (is_monophonic(ref_note_array) and not is_monophonic(pred_note_array)) or (not is_monophonic(ref_note_array) and is_monophonic(pred_note_array)):
            # add an alignment step to make reference/prediction monotonic
            raise NotImplementedError()
    
        else:
            
            pred_melody_kor_func, pred_bass_kor_func = get_kor_stream_funcs(pred_note_array)

            pred_melody_kor = pred_melody_kor_func(ref_onsets)
            pred_bass_kor = pred_bass_kor_func(ref_onsets)
            pred_ratio_kor = pred_melody_kor / pred_bass_kor

            # compare reference and prediction
            corr_melody = np.corrcoef(ref_melody_kor, pred_melody_kor)[0, 1]
            corr_bass = np.corrcoef(ref_bass_kor, pred_bass_kor)[0, 1]
            corr_ratio = np.nan if np.isnan(corr_bass) else np.corrcoef(ref_ratio_kor, pred_ratio_kor)[0, 1]

            metrics[i]["melody_kor_corr"] = corr_melody
            metrics[i]["bass_kor_corr"] = corr_bass
            metrics[i]["ratio_kor_corr"] = corr_ratio

    return metrics
