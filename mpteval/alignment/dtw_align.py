#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module provides an alignment function that aligns to chordified performance representations using DTW."""

import numpy as np
import matplotlib.pyplot as plt

from parangonar.dp.dtw import WDTW
from parangonar.dp.metrics import cdist_local


def save_matrix_plot(matrix, out_path, pred_align=None):
    plt.imshow(matrix, aspect='auto')
    if pred_align is not None:
        plt.plot(pred_align[:, 1], pred_align[:, 0], color='green')
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()

def align_chordified(p1, p2, dist_metric, out_dir, p1id, p2id,directional_weights=np.array([1, 2, 1])):
    """
    Align two performances using DTW by chordifying them into onsetwise representations.

    Parameters
    ----------
    p1 : list of PerformedChords as Dicts
        A chordified representation of a performance as a list of dicts with keys 'pc_set', 'pitch_set', 'top_pitch', 'chord_onset', 'chord_onset_norm', 'nids'
    p2 : list of PerformedChords as Dicts
        A chordified representation of a performance as a list of dicts with keys 'pc_set', 'pitch_set', 'top_pitch', 'chord_onset', 'chord_onset_norm', 'nids'
        
    dist_metric : Callable
        The distance metric to use for DTW
    directional_weights : np.array
        The weights for each of the (horizontal, diagonal, vertical) path directions for DTW. Defaults to [1,2,1] (all directions are equally costly).
        
    out_dir : pathlib.Path
        The directory path to save the predicted alignment as a .npy and the accumulated distance and distance matrices (once with, once without predicted alignment path).
    p1id : str
        The id of the p1, used in the file name for saving the outputs.
    p2id : str
        The id of the p2, used in the file name for saving the outputs.
    """
    # compute distance matrix
    pairwise_dist_m = cdist_local(p1, p2, metric=dist_metric)

    # init matcher and get alignment path
    matcher = WDTW(directional_weights)
    pred_align_path, acc_dist, cost = matcher.from_distance_matrix(
        pairwise_dist_m, 
        return_matrices=True, 
        return_cost=True
    )
    
    np.save(out_dir / f'{p1id}_{p2id}_pred_align.npy', pred_align_path)
    # loaded_matrix = np.load('output_matrix.npy')
    
    save_matrix_plot(acc_dist, out_path = out_dir / f'{p1id}_{p2id}_acc_dist.png')
    save_matrix_plot(acc_dist, out_path = out_dir / f'{p1id}_{p2id}_acc_dist_pred.png', pred_align=pred_align_path)
    save_matrix_plot(pairwise_dist_m, out_path = out_dir / f'{p1id}_{p2id}_pairw_dist.png')
    
    # stats apart from cost
    align_len = pred_align_path.shape[0]
    p1_len = len(p1)
    p2_len = len(p2)
    
    return p1id, p2id, cost, align_len, p1_len, p2_len

def process_pair_wrapper(pair, pid_to_chords_dict, dist_metric, out_dir, directional_weights):
    """
    Wrapper to align a single pair of performances, and return the results

    Parameters
    ----------
    pair : list[tuple[str, str]]
        List of 2-tuples, where each tuple contains (performance1_id, performance2_id) pairs to align/compare via chord sequences.
    pid_to_chords_dict : dict
        A dict mapping performance_id to its chord_list representation.
        
    dist_metric : Callable
        The distance metric to use for DTW
    directional_weights : np.array
        The weights for each of the (horizontal, diagonal, vertical) path directions for DTW. Defaults to [1,2,1] (all directions are equally costly).
        
    out_dir : pathlib.Path
        The directory path to save the predicted alignment as a .npy and the accumulated distance and distance matrices (once with, once without predicted alignment path).
    """
    
    p1id, p2id = pair    
    p1 = pid_to_chords_dict[str(p1id)]
    p2 = pid_to_chords_dict[str(p2id)]
    
    p1id, p2id, cost, align_len, p1_len, p2_len = align_chordified(p1, p2, dist_metric, out_dir, p1id, p2id,directional_weights)
    
    return (p1id, p2id, cost, align_len, p1_len, p2_len)