# Release Notes

## Version 0.1.4
This version introduces the following changes:

### Added
- added benchmark metrics from mir_eval for standard frame and notewise transcription evaluation, with an added option to compute notewise metrics considering only onset and velocity (ignoring offsets)

### Changed
- updated partitura dependency to 1.7.0
- `mpteval.timing.timing_metrics_from_perf()` returns only correlation metrics for melody, accompaniment and their ratio, and takes as additional argument `include_distance` to specify whether to compute DTW or histogram distance metrics (defaults to None)

### Fixed
- made timing, articulation and dynamics metrics robust to monophonic reference and polyphonic prediction.

This version was used to measure how variations and shifts in both sound characteristics and musical distribution impact the performance and robustness of several state-of-the-art piano transcription models, as described in [article in press](link).

## Version 0.1.3

This version contains the initial implementation of musically informed evaluation metrics for transcriptions of piano performances as described [here](https://arxiv.org/pdf/2406.08454). The metrics are defined to measure similarity between a predicted performance and a reference performance in terms of timing, articulation, dynamics, and harmony.
