#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module provides preprocessing fns, including:
    . chordify_perf_note_array : groups near-simultaneous note onsets into chords.
    . normalize_chord_onset_times : scales chord onset times to normalized time window [0, 1].    
"""

import numpy as np

from typing import List, Dict, Union, Optional


class PerformedChord(object):
    """
    Represents a group of near-simultaneous notes forming a chord in a performance.

    Parameters
    ----------
    pitch : List[int]
        MIDI pitch values of notes in the chord.
    ponsets : List[float]
        Onset times (seconds) of notes in the chord.
    pduration : List[float]
        Durations (seconds) of notes in the chord.
    velocities : List[float]
        MIDI velocities of notes in the chord.
    ids : List[str]
        Note IDs.
    chord_id : str
        Unique identifier for this chord.
    max_threshold : float
        Max time span (seconds) from first to last note onset in the chord.
    ioi_threshold : float
        Max inter-onset interval (seconds) between consecutive notes to be grouped.
    """

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
        self.onset_mean_norm: Optional[float] = None  # set by normalize_chord_onset_times

    @property
    def onset_start(self) -> float:
        """Onset time of the first note in the chord."""
        return np.min(self.ponsets)

    @property
    def onset_end(self) -> float:
        """Onset time of the last note in the chord."""
        return np.max(self.ponsets)

    @property
    def onset_dur(self) -> float:
        """Time span from first to last note onset."""
        return self.onset_end - self.onset_start

    @property
    def onset_mean(self) -> float:
        """Mean onset time of all notes in the chord."""
        return np.mean(self.ponsets)

    def __len__(self) -> int:
        return len(self.pitch)

    def check(self) -> bool:
        """Check that all note onsets are >= onset_start."""
        return all([onset >= self.onset_start for onset in self.ponsets])

    def add(self, pitch: int, onset: float, duration: float, velocity: float, nid: str) -> bool:
        """
        Attempt to add a note to this chord.

        Parameters
        ----------
        pitch : int
            MIDI pitch of the note.
        onset : float
            Onset time in seconds.
        duration : float
            Duration in seconds.
        velocity : float
            MIDI velocity.
        nid : str
            Note ID.

        Returns
        -------
        bool
            True if the note was added, False if it falls outside the thresholds.
        """
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

    def __str__(self) -> str:
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


def normalize_chord_onset_times(
    chords: List[PerformedChord],
    total_duration: float,
    method: str = "linear",
    **kwargs,
) -> List[PerformedChord]:
    """
    Normalize chord onset times and store result in each chord's `onset_mean_norm`.

    Parameters
    ----------
    chords : List[PerformedChord]
        List of performed chords.
    total_duration : float
        Total duration of the performance in seconds.
    method : str
        Normalization method: 'linear', 'sigmoid', 'quadratic', or 'windowed'.
    **kwargs
        Additional arguments for specific methods (e.g. `k` for sigmoid).

    Returns
    -------
    List[PerformedChord]
        Chords with `onset_mean_norm` set.
    """
    window = 0.05 * total_duration
    onsets = np.array([c.onset_mean for c in chords])

    if method == "linear":
        for c in chords:
            c.onset_mean_norm = c.onset_mean / total_duration
        return chords

    elif method == "sigmoid":
        k = kwargs.get("k", 4)
        for c in chords:
            tau = c.onset_mean / total_duration  # BUG FIX: was c.onset (no such attribute)
            c.onset_mean_norm = 1 / (1 + np.exp(-k * (tau - 0.5)))
        return chords

    elif method == "quadratic":
        for c in chords:
            tau = c.onset_mean / total_duration  # BUG FIX: was c.onset (no such attribute)
            c.onset_mean_norm = tau**2
        return chords

    elif method == "windowed":
        for c in chords:
            t = c.onset_mean
            w = np.sum(np.abs(onsets - t) <= window)
            tau = t / total_duration
            c.onset_mean_norm = tau / (1 + w)
        return chords

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Choose from: linear, sigmoid, quadratic, windowed.")


def chordify_perf_note_array(
    note_array: np.ndarray,
    ioi_threshold: float = 0.03,
    max_threshold: float = 0.05,
    normalize_onset_time: Union[str, None] = 'linear',
    return_list_of_dicts: bool = False,
) -> Union[List[PerformedChord], List[Dict]]:
    """
    Chordify a performance note array.

    Parameters
    ----------
    note_array : np.ndarray
        An input note array with fields: onset_sec, duration_sec, pitch, velocity, id.
    ioi_threshold : float, optional
        Max inter-onset interval (seconds) between consecutive notes to be grouped
        into the same chord, by default 0.03.
    max_threshold : float, optional
        Max total time span (seconds) from first to last note onset in a chord,
        by default 0.05.
    normalize_onset_time : str or None, optional
        Normalize chord onsets using this method ('linear', 'sigmoid', 'quadratic',
        'windowed'). If None, no normalization is applied. Defaults to 'linear'.
    return_list_of_dicts : bool, optional
        If True, return chords as dicts with keys: 'pc_set', 'pitch_set', 'top_pitch',
        'chord_onset', 'chord_onset_norm', 'nids'. Defaults to False.

    Returns
    -------
    List[PerformedChord] or List[Dict]
        List of performed chords, either as PerformedChord objects or dicts.
    """
    sort_idx = note_array["onset_sec"].argsort()
    note_array = note_array[sort_idx]
    note_offs = note_array['onset_sec'] + note_array['duration_sec']
    total_dur_sec = np.max(note_offs)

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

    if normalize_onset_time:
        chords = normalize_chord_onset_times(chords, total_dur_sec, normalize_onset_time)

    if return_list_of_dicts:
        chords_list = []
        for c in chords:
            c_dict = dict(
                pc_set=tuple(sorted([p % 12 for p in c.pitch])),
                pitch_set=tuple(c.pitch),
                top_pitch=max(c.pitch),
                chord_onset=c.onset_mean,
                chord_onset_norm=c.onset_mean_norm if normalize_onset_time else c.onset_mean,
                nids=c.ids,
            )
            chords_list.append(c_dict)
        return chords_list

    return chords