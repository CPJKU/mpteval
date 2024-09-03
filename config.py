import os

project_path = os.path.dirname(os.path.abspath(__file__))

eval_metrics = os.path.join(project_path, "eval")
peamt = os.path.join(eval_metrics, "peamt")
helpers = os.path.join(project_path, "helpers")
results = os.path.join(project_path, "results")
paper = os.path.join(project_path, "paper")
figs = os.path.join(paper, "figs")
tables = os.path.join(paper, "tables")
data_path = os.path.join(project_path, "data")
maestro_subset = os.path.join(data_path, "maestro_subset")
revnoise = os.path.join(data_path, "revnoise")

for dir in [peamt, figs, tables, maestro_subset, revnoise]:
    if not os.path.exists(dir):
        os.makedirs(dir)

EVAL_DATA_PATHS = {
    "maestro_subset": {
        "meta": os.path.join(data_path, "meta_csv/meta_maestro_subset.csv"),
        "data_path": os.path.join(data_path, "maestro_subset"),
    },
    "revnoise": {
        "meta": os.path.join(data_path, "meta_csv/meta_revnoise_subset.csv"),
        "data_path": os.path.join(data_path, "revnoise"),
    },
}
