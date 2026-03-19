import pandas as pd
import numpy as np

from itertools import product

from .cluster import hierarchical_structural_clustering
from .cluster_eval import evaluate_clustering


def grid_search_clustering(
    align_df,
    true_groups,
    methods=["ward", "average", "complete", "single"],
    cost_weights=[0.3, 0.5, 0.7, 1, 2, 3],
    stretch_opt_weight=[0.3, 0.5, 0.7, 1, 2, 3],
    stretch_avg_weight=[0.3, 0.5, 0.7, 1, 2, 3],
    length_weights=[0.3, 0.5, 0.7, 1, 2, 3],
    dist_thresh=[0.6, 0.7, 0.8, 0.9],
    sort_results_by_metric="adjusted_mutual_info",
):
    """
    Test all combinations of parameters and find the best.
    """
    results = []

    total_combinations = (
        len(methods)
        * len(cost_weights)
        * len(stretch_opt_weight)
        * len(stretch_avg_weight)
        * len(length_weights)
        * len(dist_thresh)
    )

    from tqdm import tqdm

    combinations = list(
        product(
            methods,
            cost_weights,
            stretch_opt_weight,
            stretch_avg_weight,
            length_weights,
            dist_thresh,
        )
    )

    combinations = [
        combo
        for combo in combinations
        if not (combo[1] == 0 and combo[2] == 0 and combo[3] == 0 and combo[4] == 0)
    ]
    print(f"Testing {len(combinations)} parameter combinations...")

    for method, c_w, so_w, sa_w, l_w, t in tqdm(combinations, desc="Grid search"):
        try:
            # Run clustering with these parameters
            predicted_groups, _, _ = hierarchical_structural_clustering(
                align_df,
                method=method,
                cost_weight=c_w,
                stretch_opt_weight=so_w,
                stretch_avg_weight=sa_w,
                length_ratio_weight=l_w,
                dist_thresh=t,
            )

            # Debug: check predicted groups
            if len(predicted_groups) == 0:
                print(
                    f"  WARNING: No groups predicted for {method}, weights=({c_w},{so_w},{sa_w},{l_w}), dist={t}"
                )
                continue

            # Evaluate
            metrics = evaluate_clustering(true_groups, predicted_groups)

            # Check if evaluation succeeded
            if np.isnan(metrics["adjusted_rand_index"]):
                print(
                    f"  WARNING: Evaluation failed for {method}, weights=({c_w},{so_w},{l_w})"
                )
                continue

            result = {
                "method": method,
                "cost_weight": c_w,
                "stretch_opt_weight": so_w,
                "stretch_avg_weight": sa_w,
                "length_weight": l_w,
                "dist_thresh": t,
                "n_predicted_groups": len(predicted_groups),
                **metrics,
            }
            results.append(result)

        except Exception as e:
            print(
                f"  FAILED: method={method}, weights=({c_w},{so_w},{sa_w},{l_w}), distance={t}: {e}"
            )
            continue

    if len(results) == 0:
        raise ValueError(
            "No successful clustering runs! Check your data and parameters."
        )

    # Sort by adjusted_rand_index
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(sort_results_by_metric, ascending=False)

    return results_df
