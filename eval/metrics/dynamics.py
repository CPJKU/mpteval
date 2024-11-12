"""
Dynamics based metrics for transcription
"""
import numpy as np
from typing import List, Tuple, Callable, Union

import partitura as pt
from partitura.performance import PerformedPart, Performance

import matplotlib.pyplot as plt

from partitura.utils.generic import interp1d


class PerformedChord(object):
    def __init__(
        self,
        pitch: List[int],
        ponsets: List[float],
        pduration: List[float],
        velocities: List[float],
        ids: List[str],
        chord_id: str,
        max_threshold: float = 0.1,
        ioi_threshold: float = 0.02,
    ) -> None:

        self.pitch = pitch
        self.ponsets = ponsets
        self.pduration = pduration
        self.velocities = velocities
        self.ids = ids
        self.max_threshold = max_threshold
        self.ioi_threshold = ioi_threshold
        self.cid = chord_id

    @property
    def onset_start(self) -> float:
        return np.min(self.ponsets)

    @property
    def onset_end(self) -> float:
        return np.max(self.ponsets)

    @property
    def onset_dur(self) -> float:
        return self.onset_end - self.onset_start

    @property
    def onset_mean(self) -> float:
        return np.mean(self.ponsets)

    def __len__(self) -> int:
        return len(self.pitch)

    def check(self):

        onset_start_crit = all([onset >= self.onset_start for onset in self.ponsets])

        return onset_start_crit

    def add(self, pitch, onset, duration, velocity, nid) -> bool:

        assert onset >= self.onset_start and onset >= self.onset_end

        if (onset - self.onset_start) <= self.max_threshold and (
            onset - self.onset_end
        ) <= self.ioi_threshold:
            self.pitch.append(pitch)
            self.ponsets.append(onset)
            self.pduration.append(duration)
            self.velocities.append(velocity)
            self.ids.append(nid)

            return True

        else:
            return False

    def __str__(self):
        out_str = (
            f"\nPerformedChord {self.cid}\n"
            f"\tonset_start: {self.onset_start:.3f}"
            f"\tonset_end: {self.onset_end:.3f}"
            f"\tonset_duration: {self.onset_dur:.3f}\n"
            "\tNotes:\n"
        )

        out_str += "\n".join(
            [
                f"\t\t{nid}, {p}, {on:.3f}, {dur:.3f}"
                for nid, p, on, dur in zip(
                    self.ids, self.pitch, self.ponsets, self.pduration
                )
            ]
        )

        return out_str


def midi_vel_to_rms(vel: np.ndarray, r_b: float = 44.0) -> np.ndarray:
    """
    Transform MIDI velocity to RMS values using the method in [1]_

    Parameters
    ----------
    vel: np.ndarray
        MIDI velocity
    r_b: float
        Desired dynamic range in dB.

    Returns
    -------
    rms: np.ndarray
        RMS energy

    References
    ----------
    .. [1] Roger B. Dannenberg (2006) The Interpretation of MIDI Velocity
           in Proceedings of the 2006 International Computer Music Conference,
           San Francisco, CA. pp. 193-196.
    """

    # See equations 7-11 in the paper
    r = 10 ** (r_b / 20)
    b = 127 / (126 * np.sqrt(r)) - 1 / 126
    m = (1 - b) / 127
    rms = (m * vel + b) ** 2

    return rms


def dynamic_range_in_db(
    vel1: np.ndarray,
    vel2: np.ndarray,
    r_b: float = 44.0,
) -> np.ndarray:
    """
    Compute the dynamic range between two MIDI velocity profiles.

    Parameters
    ----------
    vel1 : np.ndarray
        A MIDI velocity curve (a MIDI velocity for each onset)

    vel2: np.ndarray
        A MIDI velocity curve (a MIDI velocity for each onset)

    Returns
    -------
    range_in_db : np.ndarray
        A curve with the dynamic range for each onset.
    """

    rms1 = midi_vel_to_rms(vel=vel1, r_b=r_b)
    rms2 = midi_vel_to_rms(vel=vel2, r_b=r_b)
    range_in_db = 20 * np.log10(rms1 / rms2)

    return range_in_db


def get_upper_lower_stream_dynamic_range(
    note_array: np.ndarray,
    r_b: float = 44.0,
    ioi_threshold: float = 0.03,
    max_threshold: float = 0.05,
) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:

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

    upper_voice = []
    lower_voice = []

    threshold = 60
    for chord in chords:

        max_pitch_idx = np.argmax(chord.pitch)
        min_pitch_idx = np.argmin(chord.pitch)

        max_note = note_array[np.where(note_array["id"] == chord.ids[max_pitch_idx])]
        min_note = note_array[np.where(note_array["id"] == chord.ids[min_pitch_idx])]

        if chord.pitch[max_pitch_idx] > threshold:
            upper_voice.append(max_note)

        if max_pitch_idx != min_pitch_idx:

            lower_voice.append(min_note)

    upper_voice = np.vstack(upper_voice).flatten()

    lower_voice = np.vstack(lower_voice).flatten()

    upper_voice_vel_func = interp1d(
        x=upper_voice["onset_sec"],
        y=upper_voice["velocity"],
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=1e-6,
    )

    lower_voice_vel_func = interp1d(
        x=lower_voice["onset_sec"],
        y=lower_voice["velocity"],
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=1e-6,
    )

    vel_upper = upper_voice_vel_func(note_array["onset_sec"])
    vel_lower = lower_voice_vel_func(note_array["onset_sec"])

    dynamic_range = dynamic_range_in_db(
        vel1=vel_upper,
        vel2=vel_lower,
        r_b=r_b,
    )

    dynamic_range_func = interp1d(
        x=note_array["onset_sec"],
        y=dynamic_range,
        dtype=float,
        kind="previous",
        bounds_error=False,
        fill_value=-r_b,
    )

    return dynamic_range_func, upper_voice_vel_func, lower_voice_vel_func


def dynamics_metrics_from_perf(
    ref_perf: Union[PerformedPart, Performance],
    pred_perf: Union[PerformedPart, Performance],
    use_true_note_offs: bool = True,
    r_b: float = 44.0,
) -> float:

    # Use true note offs
    if use_true_note_offs:

        for ppart in ref_perf:
            ppart.sustain_pedal_threshold = 127

        for ppart in pred_perf:
            ppart.sustain_pedal_threshold = 127

    # Create reference and predicted note arrays and compute dynamic range functions
    ref_note_array = ref_perf.note_array()
    pred_note_array = pred_perf.note_array()

    ref_dynamic_range_func, _, _ = get_upper_lower_stream_dynamic_range(
        note_array=ref_note_array,
        r_b=r_b,
    )

    pred_dynamic_range_func, _, _ = get_upper_lower_stream_dynamic_range(
        note_array=pred_note_array,
        r_b=r_b,
    )

    # Compute correlation between reference and predicted dynamic range functions
    corr = np.corrcoef(
        ref_dynamic_range_func(ref_note_array["onset_sec"]),
        pred_dynamic_range_func(ref_note_array["onset_sec"]),
    )[0, 1]

    return corr
