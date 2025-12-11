#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides clustering functions to group different transcriptions 
via their pairwise alignment cost into clusters associated to different structural realisations.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


def compute_pairwise_features(align_df):
    """
        Compute (dis)similarity features that indicate structural similarity:

            . cost_matrix: the cost of the alignment path, normalized by path length
                        . average cost per alignment step
                        . the greater the poorer the alignment
            . stretch_opt_matrix: measures temporal warping compared to optimum path:
                        . how much the alignment path inflates compared to optimum path length
                        . an optimum alignment between two seqs of len N and M would have path length max(N, M)
                        . the greater the poorer the alignment
            . stretch_avg_matrix: measures temporal warping compared to average sequence length

            . length_ratio_matrix: measures similarity by comparing the two aligned sequences' length
                        . independent of alignment cost, just measures sequence length (dis)similarity
                        . range [0,1], the closer to 1 the better / structurally closer the two sequences

    Parameters
    ----------
    align_df : pd.DataFrame
    A dataframe containing the columns p1, p2, cost, align_len, p1_len, p2_len, where
        . p1, p2            ids of the two performances being aligned
        . cost              cost of their alignment
        . align_len         length of their alignment path
        . p1_len, p2_len    sequence lengths of p1 and p2
    """

    # Create distance matrix
    items = sorted(list(set(align_df["p1"]).union(set(align_df["p2"]))))
    N = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}

    # Initialize feature matrices
    cost_matrix = np.full((N, N), np.inf)
    stretch_opt_matrix = np.full((N, N), np.inf)
    stretch_avg_matrix = np.full((N, N), np.inf)
    length_ratio_matrix = np.full((N, N), 0.0)

    for _, row in align_df.iterrows():
        i = item_to_idx[row["p1"]]
        j = item_to_idx[row["p2"]]

        # Normalized cost
        cost_matrix[i, j] = cost_matrix[j, i] = row["cost"] / row["align_len"]

        # Stretch compared to optimum
        min_path_len = max(row["p1_len"], row["p2_len"])
        stretch = (row["align_len"] - min_path_len) / min_path_len
        stretch_opt_matrix[i, j] = stretch_opt_matrix[j, i] = stretch

        # Stretch compared to average
        avg_len = (row["p1_len"] + row["p2_len"]) / 2
        stretch = (row["align_len"] - avg_len) / avg_len
        stretch_avg_matrix[i, j] = stretch_avg_matrix[j, i] = stretch

        # Length ratio (similar lengths = structural correspondence)
        length_ratio = min(row["p1_len"], row["p2_len"]) / max(
            row["p1_len"], row["p2_len"]
        )
        length_ratio_matrix[i, j] = length_ratio_matrix[j, i] = length_ratio

    np.fill_diagonal(cost_matrix, 0)
    np.fill_diagonal(stretch_opt_matrix, 0)
    np.fill_diagonal(stretch_avg_matrix, 0)
    np.fill_diagonal(length_ratio_matrix, 1)

    return (
        items,
        cost_matrix,
        stretch_opt_matrix,
        stretch_avg_matrix,
        length_ratio_matrix,
    )


def hierarchical_structural_clustering(
    align_df,
    method,
    cost_weight,
    stretch_opt_weight,
    stretch_avg_weight,
    length_ratio_weight,
    dist_thresh,
    n_clusters=None,
    plot=None,
):
    """
    Perform hierarchical clustering to find structurally corresponding groups.

    Parameters
    ----------
    align_df : pd.DataFrame
    A dataframe containing the columns [p1, p2, cost, align_len, p1_len, p2_len], from which the the cost, stretch and length_ratio matrices are computed, which are used as input features for clustering.

    method : str
    The clustering method to use, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    cost_weight : float | int
    stretch_opt_weight : float | int
    stretch_avg_weight : float | int
    length_ratio_weight : float | int
    The weights to apply on the cost / stretch / length_ratio matrices.

    n_clusters: int, optional
    If given, forms n_clusters clusters, Otherwise, forms flat clusters so that between-cluster distances is no greater than 0.7 (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html)

    plot : str or None, optional
    Path to save dendrogram plot. If None (default), no plot is generated.
    """

    items, cost_mat, stretch_opt_mat, stretch_avg_mat, length_ratio_mat = (
        compute_pairwise_features(align_df)
    )

    # Combine all distance matrices
    distance_matrix = (
        (cost_mat / cost_mat.max()) * cost_weight
        + (stretch_opt_mat / (stretch_opt_mat.max() + 1e-10)) * stretch_opt_weight
        + (stretch_avg_mat / (stretch_avg_mat.max() + 1e-10)) * stretch_avg_weight
        + (1 - length_ratio_mat) * length_ratio_weight
    ) / (cost_weight + stretch_opt_weight + stretch_avg_weight + length_ratio_weight)

    # force perfect symmetry by averaging with transpose
    asymmetry = np.abs(distance_matrix - distance_matrix.T).max()
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # convert to condensed distance matrix
    condensed_dist = squareform(distance_matrix)

    # Hierarchical clustering
    Z = linkage(condensed_dist, method=method, optimal_ordering=True)

    # Plot dendrogram
    if plot:
        plt.figure(figsize=(10, 4))
        dendrogram(Z, labels=items, leaf_rotation=90)
        plt.title("Structural Similarity Dendrogram")
        plt.xlabel("Transcription")
        plt.ylabel("Distance")
        plt.tight_layout()
        if isinstance(plot, str) or isinstance(plot, Path):
            plt.savefig(plot, dpi=300)
            plt.show()
            plt.close()

    # cut tree at different heights to get clusters
    labels = fcluster(Z, t=dist_thresh, criterion="distance")
    if n_clusters:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    groups = {}
    for label in set(labels):
        groups[f"group_{label}"] = [
            items[i] for i, l in enumerate(labels) if l == label
        ]

    return groups, Z, distance_matrix


def plot_distance_matrix(items, distance_matrix, title, groups=None, out=None, figsize=(12, 10)):
    """
    Visualize similarity feature matrix, with groups highlighted if groups are not None.

    Parameters
    ----------
    align_df : pd.DataFrame
    A dataframe containing the columns [p1, p2, cost, align_len, p1_len, p2_len], from which 
    the cost, stretch and length_ratio matrices are computed, which are used as input features for clustering.

    groups : dict, optional
    A predicted or true grouping structure, structured like: {'group_1': [p1id, p3id], 'group_2': [p2id, p4id], ...}.

    """

    plt.figure(figsize=figsize)

    # reorder by groups if provided
    if groups:
        ordered_items = []
        for group in groups.values():
            ordered_items.extend(group)
        # reorder matrix
        indices = [items.index(item) for item in ordered_items]
        distance_matrix = distance_matrix[np.ix_(indices, indices)] # reorder rows and columns
        items = ordered_items

    plt.imshow(distance_matrix, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(label="Distance Matrix")
    plt.title(title)

    plt.xticks(range(len(items)), items, rotation=90, fontsize=8)
    plt.yticks(range(len(items)), items, fontsize=8)

    # draw group boundaries
    if groups:
        pos = 0
        for group in groups.values():
            pos += len(group)
            plt.axhline(pos - 0.5, color="blue", linewidth=2)
            plt.axvline(pos - 0.5, color="blue", linewidth=2)

    plt.tight_layout()
    if isinstance(out, str) or isinstance(out, Path):
        plt.savefig(out, dpi=150)
        plt.close()

def plot_n_dist_matrices(items_list, matrices, groups_dicts, titles, out=None):
    """
    Plot n distance matrices side by side with different group structures.

    Parameters
    ----------
    items_list : list of lists
        List of item lists (one per matrix)
    matrices : list of np.ndarray
        List of distance matrices
    groups_dicts : list of dict
        List of group dictionaries, e.g. [{'predicted': ...}, {'pseudo_labels': ...}, {'true': ...}]
    titles : list of str
        List of titles (one per matrix)

    All lists must be equal in length.
    """
    
    assert len(matrices) == len(items_list) == len(groups_dicts), \
        "items_list, matrices, and groups_dicts must have same length"

    n = len(matrices)

    
    fig, axes = plt.subplots(1, n, figsize=(8 * n + 1, 8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for i, (ax, items, matrix, groups_dict, title) in enumerate(
        zip(axes, items_list, matrices, groups_dicts, titles)
    ):
        # If there is no grouping for this matrix, fall back to original order
        if groups_dict:
            # Use exactly one group structure per subplot:
            # take the first (key, value) pair in this dict
            group_title, group_dict = next(iter(groups_dict.items()))

            # Flatten groups into an ordered item list
            ordered_items = []
            for group in group_dict.values():
                ordered_items.extend(group)

            # Reorder matrix according to ordered_items
            indices = [items.index(item) for item in ordered_items]
            matrix = matrix[np.ix_(indices, indices)]
            items = ordered_items
        else:
            group_title = title
            group_dict = {}

        # Plot matrix
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        # if i == n - 1:
        #     plt.colorbar(im, ax=ax, label="Distance")

        ax.set_title(title.title(), fontsize=16)

        # Set ticks (Matplotlib >= 3.6)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(items, rotation=90, fontsize=8)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(items, fontsize=8)

        # Draw group boundaries if groups exist
        if group_dict:
            pos = 0
            for group in group_dict.values():
                pos += len(group)
                ax.axhline(pos - 0.5, color="blue", linewidth=2)
                ax.axvline(pos - 0.5, color="blue", linewidth=2)


    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.02)
    cbar.set_label("Distance")

    if out is not None:
        plt.savefig(out, dpi=300, bbox_inches="tight")
    else:
        plt.show()
