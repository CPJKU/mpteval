Release Notes
=============

Version 1.1.0
=============

This version introduces the following changes:

Changed
-------

- in the frame-based majority metric `mpteval.metrics.majority` added an additional step to smoothen out spuriously (by majority) introduced note events by thresholding notes to have need a minimal duration of 50ms and merging fragmented notes (reonset within 30ms) into one long note. Also changed the majority voting to be based on dilated (+- 15 frames) roles to consider for timing jitter introduced by transcription.

Version 1.0.1
=============

This version introduces the following changes:

Changed

Added
-----

- preprocessing:
  added chordification and time normalisation fns in `mpteval.preprocessing.preprocess`
- alignment:
  - added alignment function and wrapper for parallelized processing to align chordified performances via DTW in `mpteval.alignment.dtw_align`
  - added custom distance metric in `mpteval.alignment.dtw_dist`
- clustering:
  - added clustering fns in `mpteval.clustering.cluster` to cluster transcriptions of the same piece into different performed structural realisations
  - added cluster evaluation in `mpteval.clustering.cluster_eval`
  - added grid search for clustering parameters in `mpteval.clustering.cluster_gs`
- metrics:
  - added majority based metric in `mpteval.metrics.majority`: given multiple transcriptions and the source audio file, selects a base transcription by audio duration proximity, removes notes found in only a minority of transcriptions (likely false positives), and inserts notes found in the majority but missing from the base (likely negatives).
- utils:
  - added utility fns in `mpteval.utils`:
    - `MusicEncoder`: JSON encoder for numpy types
    - `plot_piano_roll`: Visualize piano rolls
    - `create_note_list`: Convert note arrays to structured lists with onset, offset, pitch, velocity
    - `is_monophonic`: Check if a note array is monophonic
    - `midi_to_piano_note`: Convert MIDI numbers to human-readable note names
    - `alignment_list_to_df`: Convert alignment lists to pandas DataFrames
    - `group_by_structure`: Group performance IDs by structural features
    - `pairwise_distance_matrix`: Compute pairwise distance matrices between sequences
    - `accumulated_cost_matrix`: Compute DTW accumulated cost matrix
    - `optimal_warping_path`: Extract optimal warping path from accumulated cost matrix
    - `dynamic_time_warping`: Vanilla dynamic time warping implementation
    - `fast_dynamic_time_warping`: Fast approximate DTW using fastdtw library
    - `greedy_note_alignment`: Extract note-level alignments from DTW warping paths
    - `notewise_alignment`: Complete note-wise music alignment combining DTW with greedy matching

Changed
-------

- moved all (global) configs to `config.py`
- renamed `mpteval.benchmarks` to `mpteval.metrics.objective`
- use consistent voice separation function in `mpteval.metrics_feature.timing` as in `mpteval.metrics_feature.articulation`

Version 0.1.4
=============

This version was used to measure how variations and shifts in both sound characteristics and musical distribution impact the robustness of several state-of-the-art piano transcription models, as described [here](https://link.springer.com/article/10.1186/s13636-025-00428-z).
It introduced the following changes:

Added
-----

- added benchmark metrics from mir_eval for standard frame and notewise transcription evaluation, with an added option to compute notewise metrics considering only onset and velocity (ignoring offsets)

Changed
-------

- updated partitura dependency to 1.7.0
- `mpteval.timing.timing_metrics_from_perf()` returns only correlation metrics for melody, accompaniment and their ratio, and takes as additional argument `include_distance` to specify whether to compute DTW or histogram distance metrics (defaults to None)

Fixed
-----

- made timing, articulation and dynamics metrics robust to monophonic reference and polyphonic prediction.

Version 0.1.3
=============

This version contains the initial implementation of musically informed evaluation metrics for transcriptions of piano performances as described [here](https://arxiv.org/pdf/2406.08454). The metrics are defined to measure similarity between a predicted performance and a reference performance in terms of timing, articulation, dynamics, and harmony.
