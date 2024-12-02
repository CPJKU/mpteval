# Towards Musically Informed Evaluation of Piano Transcription Models
This repository provides a set of evaluation metrics designed for piano transcription evaluation. The metrics are musically informed, meaning they take into account the nuances of musical performance, such as dynamics, note onset, and duration, to offer more differentiated and musically relevant assessments of transcription quality.
Note that these metrics are a work in progress and actively being developed/refined/extended. Expect future updates, and feel free to contribute or share feedback!

# Metrics computation
The following code loads a reference and a predicted MIDI and computes how well the transcription preserves timing information in the performance:
```
import mpteval
from mpteval.timing import timing_metrics_from_perf
import partitura as pt

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
- Partitura 1.6.0* (**Note** that currently only Partitura 1.5.0 is out (we're working on the next release! We've set the auto-install version of partitura to 1.5.0 in the meantime, to get the correct metrics please manually install the relevant branch from partitura using: `!pip install git+https://github.com/CPJKU/partitura.git@develop`))


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
