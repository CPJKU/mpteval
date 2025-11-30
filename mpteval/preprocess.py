#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module provides preprocessing fns, including:
    . chordify_perf_note_array : groups near-simultaneous note onsets into chords.
    . normalize_chord_onset_times : scales chord onset times to normalized time window [0, 1].    
"""

import numpy as np

from typing import List

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


def normalize_chord_onset_times(chords, total_duration, method="sigmoid", **kwargs):
    
    window = 0.05 * total_duration
    onsets = np.array([c.onset_mean for c in chords])
    
    if method == "linear":
        for c in chords:
            c.onset_mean_norm = c.onset_mean / total_duration
        return chords

    elif method == "sigmoid":
        k = kwargs.get("k", 4)
        for c in chords:
            tau = c.onset / total_duration
            c.onset_mean_norm = 1 / (1 + np.exp(-k * (tau - 0.5)))
        return chords

    elif method == "quadratic":
        for c in chords:
            tau = c.onset / total_duration
            c.onset_mean_norm = tau**2
        return chords

    elif method == "windowed":
        # compute local density inside ±5% window
        for _, c in enumerate(chords):
            t = c.onset
            # count how many other onsets fall inside the window
            w = np.sum(np.abs(onsets - t) <= window)
            
            tau = t / total_duration
            # density compression
            c.onset_mean_norm = tau / (1 + w)
        return chords


def chordify_perf_note_array(
    note_array: np.ndarray,
    ioi_threshold: float = 0.03,
    max_threshold: float = 0.05,
    normalize_onset_time = 'linear',
    return_list_of_dicts = False,
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
    normalize_onset_time : str, defaults to 'linear'
        Normalize chord onsets by normalize_onset_time method
    return_list_of_dicts: boolean, defaults to False
        Return PerformedChords as Dicts with keys 'pc_set', 'pitch_set', 'top_pitch', 'chord_onset', 'nids'
    
    Returns
    -------
    chords : List[PerformedChord] (List[Dict] if return_list_of_dicts=True)
        List of performed chords.
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
                
            c_dict = dict(pc_set=tuple(sorted([p%12 for p in c.pitch])),
                        pitch_set=tuple(c.pitch), 
                        top_pitch=max(c.pitch), 
                        chord_onset=c.onset_mean, 
                        chord_onset_norm = c.onset_mean_norm if normalize_onset_time else c.onset_mean,
                        nids=c.ids)
            chords_list.append(c_dict)
        
        return chords_list    

    return chords
    
