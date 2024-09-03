# Towards Musically Informed Evaluation of Piano Transcription Models
Piano transcription models are typically evaluted using information-retrieval metrics precision, recall and F1 score. In this project we propose a set of musically informed metrics to capture more nuanced transcription errors for expressive performance dimensions such as timing, articulation, harmony and dynamics.
 
# Requirements
- Python 3.9
- conda

# Setup
- Clone the repo and create the conda environment using the provided [`mpteval.yml`](mpteval.yml) environment file.
- Download the evaluation data [here](https://zenodo.org/records/12731999) and move it to `data/maestro_subset` and `data/revnoise`

# Metrics computation
- To analyse the results, see [`notebooks/metrics_demonstration.ipynb`](notebooks/metrics_demonstration.ipynb)
- If you want to compute the metrics from scratch, run `python main.py`. By default, this will compute *all* metrics (apart from `peamt`, see below) for the `maestro_subset` subset. You can use the `--subset` flag to choose another subset, and the `--eval_metric` flag to choose the metric you want to compute. The results will be stored in the `results` directory, as `[subset_]<metric>_<datetimestamp>.csv` (where `subset` is optional, e.g. only explicitely specified for `revnoise`)


## `peamt` metric computation
To compute the `peamt` metric, run `git submodule init` followed by `git submodule update` to download the submodule contents into `eval/peamt` folder. Then create and activate the `peamt` environment using the following:
```bash
conda create -n peamt python=3.6
conda activate peamt
cd eval/peamt
pip install . # you may ignore pip's dependency resolvers' possible complaints here
pip install pandas tqdm # as well as here
```
Copy the script [`eval/compute_peamt.py`](eval/compute_peamt.py) into the `eval/peamt/` subdirectory (git doesn't allow non-empty submodule directory initialization) and navigate back to the main director using `cd ../..`. Compute the `peamt` metric by running `python eval/peamt/compute_peamt.py`. You can again use the `--subset` flag to choose which subset you wish to evaluate on.


# Citing
If you use our evaluation set and/or metrics in your research, please cite the relevant [paper](https://arxiv.org/abs/2406.08454):

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
