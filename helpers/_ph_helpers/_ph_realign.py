import os
import numpy as np
import partitura as pt
import glob
import madmom
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.signal import convolve
from numpy.linalg import norm

from typing import Optional, Tuple

from synthesize_fluidsynth import (
    synthesize_fluidsynth,
    # save_wav_fluidsynth,
    # convert_audio_to_flac,
)

from partitura.io.importmatch import load_matchfile
from partitura.io.exportmatch import save_match

import warnings

# Ignore all warnings from a specific module
warnings.filterwarnings("ignore", module="partitura")

SAMPLE_RATE = 44100
FRAME_SIZE = 1024
HOP_SIZE = 64
# FRAME_SIZE = int(np.round(SAMPLE_RATE * 0.05))
# HOP_SIZE = int(np.round(SAMPLE_RATE * 0.0005))


def fast_cosine_similarity(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    cos_conv = convolve(X, np.flip(kernel) / norm(kernel), mode="valid")

    X_norm_est = np.sqrt(convolve(X**2, np.ones(kernel.shape), mode="valid"))

    cosine_similarity = (cos_conv / (X_norm_est)).sum(1)

    distance = 1 - cosine_similarity

    return distance


def create_windows(
    arr: np.ndarray,
    window_size: int,
    stride: int = 1,
    start: Optional[int] = None,
    end: Optional[int] = None,
    pad: bool = False,
) -> np.ndarray:
    """
    Create views into a given array corresponding to a sliding window.
    If no start or end is given, then the start/end of the given array is
    used.

    Parameters
    ----------
    arr : np.ndarray
        array to be indexed
    start : int, optional
        where to start indexing
    end : int, optional
        where to end indexing

    Returns
    -------
    windows : np.ndarray
        dim 0 is all windows, dims 1 and 2 are the actual window
        dimensions.
    """
    if start is None:
        start = 0
    if end is None:
        end = arr.shape[0]

    arr_len = end - start

    sub_window_ids = (
        start
        + np.expand_dims(np.arange(window_size), 0)
        + np.expand_dims(np.arange(end - start - window_size, step=stride), 0).T
    )

    windowed_array = arr[sub_window_ids]

    if pad and len(sub_window_ids) * stride < arr_len:
        pad_length = window_size - (arr_len - len(sub_window_ids) * stride)
        pad_window = np.pad(
            arr[-(arr_len - len(sub_window_ids) * stride) :], ((0, pad_length), (0, 0))
        )
        windowed_array = np.vstack((windowed_array, pad_window[np.newaxis]))

    return windowed_array


def generate_synths_for_testing():
    out_midi_audio_dir = "../synth_midi"
    match_dir = "../match"

    if not os.path.exists(out_midi_audio_dir):
        os.mkdir(out_midi_audio_dir)

    match_files = glob.glob(os.path.join(match_dir, "*.match"))

    for i, fn in enumerate(match_files):

        print(f"synthesizing {fn}")
        perf, alignment = pt.load_match(fn)
        for ppart in perf:
            for note in ppart.notes:
                # fix pedal info
                note["channel"] = 0
                note["track"] = 0

            for ctrl in ppart.controls:
                ctrl["channel"] = 0
                ctrl["track"] = 0
        out_fn = os.path.join(
            out_midi_audio_dir, os.path.basename(fn).replace(".match", "_synth.wav")
        )

        save_wav_fluidsynth(input_data=perf, out=out_fn)
        if i == 3:
            break


if __name__ == "__main__":

    out_midi_audio_dir = "../synth_midi"
    match_dir = "../match"
    musicxml_dir = "../scores"
    audio_dir = "../audio"
    adjusted_match_dir = "../match_adjusted"
    adjusted_midi_dir = "../midi_adjusted"

    if not os.path.exists(adjusted_midi_dir):
        os.mkdir(adjusted_midi_dir)

    window_size_in_sec = 5

    window_size = int(np.ceil(window_size_in_sec * SAMPLE_RATE / HOP_SIZE))

    match_files = glob.glob(os.path.join(match_dir, "*.match"))

    match_files.sort()

    for fn in match_files:

        print(f"Processing {os.path.basename(fn)}...")

        audio_fn = os.path.join(
            audio_dir,
            os.path.basename(fn).replace(".match", ".wav"),
        )
        # Filenames of the  adjusted MIDI and match files
        adjusted_perf_fn = os.path.join(
            adjusted_midi_dir,
            os.path.basename(fn).replace(".match", "_adj.mid"),
        )
        adjusted_match_fn = os.path.join(
            adjusted_match_dir,
            os.path.basename(fn).replace(".match", "_adj.match"),
        )

        if os.path.basename(fn) in ("kv284_3.match", "kv331_1.match"):
            continue
        if os.path.exists(adjusted_perf_fn) and os.path.exists(adjusted_match_fn):
            continue
        # load performance, alignment and score

        if os.path.basename(fn) not in ("kv331_1.match", "kv284_3.match"):
            perf, alignment = pt.load_match(fn)
            score = pt.load_score(
                os.path.join(
                    musicxml_dir,
                    os.path.basename(fn).replace(".match", ".musicxml"),
                )
            )
            assume_unfolded = False
        else:
            perf, alignment, score = pt.load_match(fn, create_score=True)
            assume_unfolded = True
        spart = score[0]

        # Set sound off to note off
        for ppart in perf:
            for note in ppart.notes:
                # fix pedal info
                note["channel"] = 0
                note["track"] = 0

            for ctrl in ppart.controls:
                ctrl["channel"] = 0
                ctrl["track"] = 0

        # synthesize performance
        synth_perf = synthesize_fluidsynth(
            note_info=perf,
            samplerate=SAMPLE_RATE,
        )

        # Get spectrogram of the synthesized performance
        synth_signal = madmom.audio.FramedSignal(
            synth_perf,
            frame_size=FRAME_SIZE,
            hop_size=HOP_SIZE,
            sample_rate=SAMPLE_RATE,
        )

        synth_spect = madmom.audio.LogarithmicFilteredSpectrogram(synth_signal)

        # Compute spectrogram of the audio recording
        audio_signal = madmom.audio.FramedSignal(
            audio_fn,
            frame_size=FRAME_SIZE,
            hop_size=HOP_SIZE,
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )

        audio_spect = madmom.audio.LogarithmicFilteredSpectrogram(audio_signal)

        # Get first onset time from the performance in the match file
        synth_frame_times = np.arange(synth_signal.num_frames) * (
            HOP_SIZE / SAMPLE_RATE
        )
        note_array = perf.note_array()
        first_onset = note_array["onset_sec"].min()
        synth_start = abs(synth_frame_times - first_onset).argmin()

        # Get non-overlapping windows of the spectrogram of the synthesized performance
        synth_frames = create_windows(
            arr=synth_spect,
            window_size=window_size,
            stride=window_size,
            start=synth_start,
        )

        # Get time of the first onset in the recording
        audio_start = int(np.ceil(0 * SAMPLE_RATE / HOP_SIZE))
        audio_end = int(np.ceil(10 * SAMPLE_RATE / HOP_SIZE))
        ssw = synth_frames[0]
        distance = fast_cosine_similarity(audio_spect[audio_start:audio_end], ssw)
        first_onset_audio = distance.argmin() * HOP_SIZE / SAMPLE_RATE
        first_onset_in_frames = distance.argmin()
        window_time = distance.argmin() * HOP_SIZE / SAMPLE_RATE

        # time shift between the synthesized performance and the recording
        # in seconds (this is the quantity that we need to shift the first)
        # window
        window_shift = window_time - first_onset

        # Start position for the spectrogram windows
        audio_time = first_onset_audio

        audio_times = [audio_time]
        window_shifts = [window_shift]

        for wix in range(1, len(synth_frames)):
            print(
                f"{os.path.basename(fn)} frame {wix + 1}/{len(synth_frames)}: {window_shift}"
            )
            
            # Estimate of the start of the window in the recording
            # given the previous window
            audio_time += window_size_in_sec

            # Start time of the window in the synthesized performance
            synth_time = first_onset + wix * window_size_in_sec

            # set start and end of the window in the recording
            # around the estimated start of the window
            audio_start_sec = audio_time - 0.5 * window_size_in_sec
            audio_end_sec = audio_time + 1.5 * window_size_in_sec
            audio_start = int(np.ceil(audio_start_sec * SAMPLE_RATE / HOP_SIZE))
            audio_end = int(np.ceil(audio_end_sec * SAMPLE_RATE / HOP_SIZE))

            # Get current window from the spectrogram of the synthesized
            # performance
            ssw = synth_frames[wix]

            # Compute the cosine distance between the window in the recording
            # and the spectrogram
            distance = fast_cosine_similarity(audio_spect[audio_start:audio_end], ssw)

            # Update the audio time according to the minimal distance
            audio_time = (distance.argmin() * HOP_SIZE / SAMPLE_RATE) + audio_start_sec

            # Compute shift
            window_shift = audio_time - synth_time
            window_shifts.append(window_shift)
            audio_times.append(audio_time)

        np.savetxt(
            os.path.basename(fn).replace(".match", "_shifts.txt"), 
            np.array(window_shifts),
        )

        # Adjust the performance
        perf_notes = perf[0].notes
        controls = perf[0].controls
        ids_moved = []
        controls_moved = []
        for i, ws in enumerate(window_shifts):

            if i == 0:
                st = 0
                et = first_onset + window_size_in_sec
            else:
                st = first_onset + i * window_size_in_sec
                et = st + window_size_in_sec

            for note in [
                n for n in perf_notes if n["note_on"] >= st and n["note_on"] < et
            ]:
                if note["id"] not in ids_moved:
                    ids_moved.append(note["id"])
                    note["note_on"] += ws
                    note["note_off"] += ws
                # else:
                #     print("already moved!")

            for ctrl in [c for c in controls if c["time"] >= st and c["time"] < et]:
                if ctrl not in controls_moved:
                    ctrl["time"] += ws
                    controls_moved.append(ctrl)

        print("Save MIDI")
        pt.save_performance_midi(
            performance_data=perf,
            out=adjusted_perf_fn,
        )

        # Save match file
        # Get original match file to get meta info
        mf = load_matchfile(fn)
        print("Save match file")
        # Save match file
        save_match(
            alignment=alignment,
            performance_data=perf,
            score_data=spart,
            out=adjusted_match_fn,
            mpq=perf[0].mpq,
            ppq=perf[0].ppq,
            piece=mf.info("piece"),
            performer=mf.info("performer"),
            composer=mf.info("composer"),
            score_filename=mf.info("scoreFileName"),
            performance_filename=os.path.basename(adjusted_perf_fn),
            assume_unfolded=assume_unfolded,
            infer_tempo_indication_from_score=False,
        )
