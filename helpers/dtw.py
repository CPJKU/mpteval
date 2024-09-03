from typing import Tuple, Union, List

import numpy as np
from fastdtw import fastdtw
from scipy.spatial import distance as sp_dist
from scipy.spatial.distance import cdist


def pairwise_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise distance matrix of two sequences

    Parameters
    ----------
    X : np.ndarray
        A 2D array with size (n_observations, n_features)
    Y : np.ndarray
        A 2D array with size (m_observations, n_features)
    metric: str
        A string defining a metric (see possibilities
        in scipy.spatial.distance.cdist)

    Returns
    -------
    C : np.ndarray
        Pairwise cost matrix
    """
    if X.ndim == 1:
        X, Y = np.atleast_2d(X, Y)
        X = X.T
        Y = Y.T
    C = cdist(X, Y, metric=metric)
    return C


def accumulated_cost_matrix(C: np.ndarray) -> np.ndarray:
    """
    Dynamic time warping cost matrix from a pairwise distance matrix

    Parameters
    ----------
    D : double array
        Pairwise distance matrix (computed e.g., with `cdist`).

    Returns
    -------
    D : np.ndarray
        Accumulated cost matrix
    """
    N = C.shape[0]
    M = C.shape[1]
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n - 1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m - 1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n - 1, m], D[n, m - 1], D[n - 1, m - 1])
    return D


def optimal_warping_path(D: np.ndarray) -> np.ndarray:
    """
    Compute the warping path given an accumulated cost matrix

    Parameters
    ----------
    D: np.ndarray
        Accumulated cost Matrix

    Returns
    -------
    P: np.ndarray
        Optimal warping path
    """
    N = D.shape[0]
    M = D.shape[1]
    n = N - 1
    m = M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m], D[n, m - 1])
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            elif val == D[n - 1, m]:
                cell = (n - 1, m)
            else:
                cell = (n, m - 1)
        P.append(cell)
        (n, m) = cell
    P.reverse()
    return np.array(P)


def dynamic_time_warping(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    return_distance: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Naive Implementation of Vanilla Dynamic Time Warping

    Parameters
    ----------
    X : np.ndarray
        Array X
    Y: np.ndarray
        Array Y
    metric: string
        Name of the metric to use. See possible metrics in
        `scipy.spatial.distance`.
    return_distance : bool
       Return the dynamic time warping distance.


    Returns
    -------
    warping_path: np.ndarray
        The warping path for the optimal alignment.
    dtwd : float
        The dynamic time warping distance of the alignment.
        This distance is only returned if `return_distance` is True.
    """
    # Compute pairwise distance matrix
    C = pairwise_distance_matrix(X, Y, metric=metric)
    # Compute accumulated cost matrix
    D = accumulated_cost_matrix(C)
    dtwd = D[-1, -1]
    # Get warping path
    warping_path = optimal_warping_path(D)

    if return_distance:
        return warping_path, dtwd
    return warping_path


def fast_dynamic_time_warping(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
    return_distance: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
     Fast Dynamic Time Warping

    This is an approximate solution to dynamic time warping.

    Parameters
    ----------
    X : np.ndarray
        Array X.
    Y: np.ndarray
        Array Y.
    metric : str
        The name of the metric to use. See possible metrics in
        `scipy.spatial.distance`.

    Returns
    -------
    warping_path: np.ndarray
        The warping path for the best alignment. The first column
        are indices in array `X` and the second column represents
        the corresponding index in array `Y`.
    dtwd : float
        The dynamic time warping distance of the alignment.
    """

    # Get distance measure from scipy dist
    dist = getattr(sp_dist, metric)
    dtwd, warping_path = fastdtw(X, Y, dist=dist)

    # Make path a numpy array
    warping_path = np.array(warping_path, dtype=int)

    if return_distance:
        return warping_path, dtwd

    return warping_path


def greedy_note_alignment(
    warping_path: np.ndarray,
    idx1: np.ndarray,
    note_array1: np.ndarray,
    idx2: np.ndarray,
    note_array2: np.ndarray,
) -> List[dict]:
    """
    Greedily find and store possible note alignments

    Parameters
    ----------
    warping_path : numpy ndarray
        alignment sequence idx in stacked columns
    idx1: numpy ndarray
        pitch, start, and end coordinates of all notes in note_array1
    note_array1: numpy structured array
        note_array of sequence 1 (the score)
    idx2: numpy ndarray
        pitch, start, and end coordinates of all notes in note_array2
    note_array2: numpy structured array
        note_array of sequence 2 (the performance)

    Returns
    ----------
    note_alignment : list
        list of note alignment dictionaries

    """
    note_alignment = []
    used_notes1 = []
    used_notes2 = []

    coord_info1 = idx1
    if idx1.shape[1] == 3:
        # Assume that the first column contains the correct MIDI pitch
        coord_info1 = np.column_stack((idx1, idx1[:, 0]))

    coord_info2 = idx2

    if idx2.shape[1] == 3:
        # Assume that the first column contains the correct MIDI pitch
        coord_info2 = np.column_stack((idx2, idx2[:, 0]))

    # loop over all notes in sequence 1
    for note1, coord1 in zip(note_array1, coord_info1):
        note1_id = note1["id"]
        pc1, s1, e1, pitch1 = coord1

        # find the coordinates of the note in the warping_path

        idx_in_warping_path = np.all(
            [warping_path[:, 0] >= s1, warping_path[:, 0] <= e1], axis=0
        )
        # print(idx_in_warping_path, idx_in_warping_path.shape)
        range_in_sequence2 = warping_path[idx_in_warping_path, 1]
        max2 = np.max(range_in_sequence2)
        min2 = np.min(range_in_sequence2)

        # loop over all notes in sequence 2 and pick the notes with same pitch
        # and position
        for note2, coord2 in zip(note_array2, coord_info2):
            note2_id = note2["id"]
            pc2, s2, e2, pitch2 = coord2
            if note2_id not in used_notes2:
                if pitch2 == pitch1 and s2 <= max2 and e2 >= min2:

                    note_alignment.append(
                        {
                            "label": "match",
                            "score_id": note1_id,
                            "performance_id": str(note2_id),
                        }
                    )
                    used_notes2.append(str(note2_id))
                    used_notes1.append(note1_id)

        # check if a note has been found for the sequence 1 note,
        # otherwise add it as deletion
        if note1_id not in used_notes1:
            note_alignment.append({"label": "deletion", "score_id": note1_id})
            used_notes1.append(note1_id)

    # check again for all notes in sequence 2, if not used,
    # add them as insertions
    for note2 in note_array2:
        note2_id = note2["id"]
        if note2_id not in used_notes2:
            note_alignment.append(
                {
                    "label": "insertion",
                    "performance_id": str(note2_id),
                }
            )
            used_notes2.append(str(note2_id))

    return note_alignment


def notewise_alignment(
    reference_features: np.ndarray,
    performance_features: np.ndarray,
    score_note_array: np.ndarray,
    performance_note_array: np.ndarray,
    sidx: np.ndarray,
    pidx: np.ndarray,
    metric: str = "euclidean",
) -> List[dict]:
    """
    Note-wise music alignment

    Parameters
    -----------
    reference_features : np.ndarray
        Features of the reference (usually the score). Usually
        a piano roll or pitch class distribution
    performance_features: np.ndarray
        Features of the performance.
    sidx : np.ndarray
        Indices of the notes in the reference_features
    pidx : np.ndarray
        Indices of the notes in the performance_features
    metric : str
        Local metric for on-line time warping

    Returns
    -------
    note_alignment : List[dict]
        List with alignment information for each note in
        the score and the performance.
    """

    # Dynamic time warping
    dtw_alignment = fast_dynamic_time_warping(
        X=reference_features,
        Y=performance_features,
        metric=metric,
    )
    # Greedy note level alignment
    note_alignment = greedy_note_alignment(
        warping_path=dtw_alignment,
        idx1=sidx,
        note_array1=score_note_array,
        idx2=pidx,
        note_array2=performance_note_array,
    )

    return note_alignment
