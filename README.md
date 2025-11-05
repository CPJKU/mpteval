# Towards Musically Informed Evaluation of Piano Transcription Models
[//]: # (<p align="center"> )

[![PyPI Package](https://img.shields.io/pypi/v/mpteval)](https://pypi.org/project/mpteval/)
[![DOI](https://zenodo.org/badge/DOI/13897703.svg)](https://doi.org/10.5281/zenodo.12731998)

This repository provides a set of evaluation metrics designed for piano transcription evaluation. The metrics are musically informed, meaning they take into account the nuances of musical performance, such as dynamics, note onset, and duration, to offer more differentiated and musically relevant assessments of transcription quality.
Note that these metrics are a work in progress and actively being developed/refined/extended. Expect future updates, and feel free to contribute or share feedback!

# Metrics computation
The following code loads a reference and a predicted MIDI and computes how well the transcription preserves timing information in the performance:

```
import mpteval
import partitura as pt

from mpteval.timing import timing_metrics_from_perf

ref_perf = pt.load_performance_midi(mpteval.REF_MID)
pred_perf = pt.load_performance_midi(mpteval.PRED_MID)

timing_metrics = timing_metrics_from_perf(ref_perf, pred_perf)
```

# Setup
The easiest way to install the package is via:
```
pip install mpteval
```

## Dependencies
- Python 3.9
- Partitura 1.7.0

# Citing
If you use our metrics in your research, please cite the relevant [paper](https://arxiv.org/abs/2406.08454):
```
@inproceedings{hu2024towards,
    title = {{Towards Musically Informed Evaluation of Piano Transcription Models}},
    author = {Hu, Patricia and Mart\'ak, Luk\'a\v{s} Samuel and Cancino-Chac\'on, Carlos and Widmer, Gerhard},
    booktitle = {{Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)}},
    year = {2024}
}
```

## Acknowledgments
This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (["Whither Music?"](https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/)).
