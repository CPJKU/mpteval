#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains a custom distance fn for DTW:
    cost(A,B) = α * cost_sym(A,B) + (1 - α) * cost_time(A.t_mean, B.t_mean) 
    where
        cost_sym measures the harmonic distance
        cost_time measures the time distance
"""

import numpy as np

def jaccard_cost(a_pc, b_pc):
    
    A, B = set(a_pc), set(b_pc)
    
    if not A and not B: return 0.0
    if not A or not B: return 1.0
    return 1.0 - (len(A & B) / len(A | B))

def jaccard_cost_with_size_penalty(a_pc, b_pc, size_weight=0.5):
    A, B = set(a_pc), set(b_pc)
    if not A and not B: return 0.0
    if not A or not B: return 1.0
    
    # standard jaccard dist
    jaccard_dist = 1.0 - (len(A & B) / len(A | B))
    
    # size difference penalty (0 if same size, 1 if completely different)
    max_size = max(len(A), len(B))
    size_penalty = abs(len(A) - len(B)) / max_size if max_size > 0 else 0
    
    return (1 - size_weight) * jaccard_dist + size_weight * size_penalty

def symmetric_difference_cost(a_pc, b_pc):
    A, B = set(a_pc), set(b_pc)
    if not A and not B: return 0.0
    if not A or not B: return 1.0
    
    # penalizes both missing elements AND extra elements
    return len(A ^ B) / len(A | B)  # symmetric difference / union

def dice_cost(a_pc, b_pc):
    A, B = set(a_pc), set(b_pc)
    if not A and not B: return 0.0
    if not A or not B: return 1.0
    
    # more sensitive to size differences than Jaccard
    return 1.0 - (2 * len(A & B) / (len(A) + len(B)))


def jaccard_melody_cost(a_pc, b_pc, k=3):
    """
    compute cost between two chords using:
      cost = 0.6*(1 - Jaccard) + 0.4*(1 - top_pitch_match).
    """
    A, B = set(a_pc), set(b_pc)

    jaccard = jaccard_cost(A, B)
    
    # top pitch (melody) similarity
    if len(A) == 0 or len(B) == 0:
        top_match = 0.0
    else:
        topA = max(A)
        topB = max(B)
        diff = abs(topA - topB)
        top_match = np.exp(-diff / k)

    cost = 0.6*(1 - jaccard) + 0.4*(1 - top_match)
    return float(cost)

def time_distance(t1, t2):
    return abs(t1 - t2)


def composite_cost(a, b, alpha=1.0, pitch_feature='pc_set', time_feature='chord_onset_norm', harmonic_metric='jaccard'):
    if harmonic_metric == 'jaccard':
        cs = jaccard_cost(a[pitch_feature], b[pitch_feature])
    elif harmonic_metric == 'jaccard_melody_cost':
        cs = jaccard_melody_cost(a[pitch_feature], b[pitch_feature])
    else:
        raise ValueError(f"Unknown harmonic_metric: '{harmonic_metric}'")
    
    ct = time_distance(a[time_feature], b[time_feature])
    return alpha * cs + (1 - alpha) * ct
