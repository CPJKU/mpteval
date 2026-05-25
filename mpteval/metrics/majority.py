import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import csc_matrix
from scipy.ndimage import binary_dilation
from itertools import groupby

def _dilate_rolls(dense, radius=15):
    """Dilate each pitch row independently along the time axis."""
    structure = np.ones((1, 2 * radius + 1), dtype=bool)
    return [binary_dilation(d.astype(bool), structure=structure).astype(int) for d in dense]

def _to_dense_binary(piano_rolls):
    """Convert sparse piano rolls to dense binary numpy arrays."""
    return [np.array(pr.todense() > 0).astype(int) for pr in piano_rolls]


def _align_truncate(dense):
    """Truncate all matrices to the shortest time dimension."""
    min_time = min(d.shape[1] for d in dense)
    aligned = [d[:, :min_time] for d in dense]
    n_rolls = len(dense)
    effective_n = np.full(min_time, n_rolls, dtype=int)          # (time,)
    effective_n_2d = np.broadcast_to(                             # (pitch, time)
        effective_n[np.newaxis, :], aligned[0].shape
    )
    return aligned, effective_n_2d

def _align_to_base(dense, base_idx):
    """
    Pad/truncate all matrices to match the base roll's time dimension.
    Returns aligned matrices and a per-frame valid-count vector.
    """
    base_time = dense[base_idx].shape[1]
    aligned, valid_masks = [], []
    for d in dense:
        if d.shape[1] < base_time:
            pad = base_time - d.shape[1]
            valid = np.ones(base_time, dtype=int)
            valid[d.shape[1]:] = 0
            d = np.pad(d, ((0, 0), (0, pad)), constant_values=0)
        else:
            d = d[:, :base_time]
            valid = np.ones(base_time, dtype=int)
        aligned.append(d)
        valid_masks.append(valid)

    effective_n = np.stack(valid_masks).sum(axis=0)           # (time,)
    effective_n_2d = np.broadcast_to(                         # (pitch, time)
        effective_n[np.newaxis, :], aligned[0].shape          # use aligned[0], not dense[0]
    )
    return aligned, effective_n_2d


def _agreement_masks(dense, effective_n_2d, minority_strict=True):
    """
    Compute per-cell agreement masks from aligned binary matrices.
    Returns (sum_active, agreement, majority, disagreement).

    If minority_strict=True then disagreement means an active frame (note) is only found in one piano roll, otherwise minority disagreement means the note is found in fewer than half piano rolls.
    """
    stacked = np.stack(dense, axis=0)       # (n_rolls, pitch, time)
    sum_active = stacked.sum(axis=0)

    minority = sum_active == 1 if minority_strict else sum_active < effective_n_2d / 2

    return (
        sum_active,
        sum_active == effective_n_2d,                                        # all agree
        (sum_active > effective_n_2d / 2) & (sum_active < effective_n_2d),  # majority
        minority,                                                             # disagreement
    )


def _build_rgb(shape, agreement, majority, disagreement):
    """Paint the standard white/green/blue/red agreement image."""
    rgb = np.ones((*shape, 3))        # white = silent
    rgb[majority]    = [0.0, 0.0, 1.0]
    rgb[agreement]   = [0.0, 0.7, 0.0]
    rgb[disagreement]= [1.0, 0.0, 0.0]
    return rgb


def _plot_roll(rgb, title='Piano Roll Comparison', legend_extra=None, n_rolls=None, out_path=None):
    """Render an RGB piano-roll image with the standard legend."""
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.imshow(rgb, aspect='auto', origin='lower')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Pitch')
    ax.set_title(title)

    majority_label = (
        f'Majority (≥{np.round(n_rolls / 2)} rolls)' if n_rolls else 'Majority (>n/2 rolls)'
    )
    legend = [
        mpatches.Patch(color='green', label='Agree (all rolls)'),
        mpatches.Patch(color='blue',  label=majority_label),
        mpatches.Patch(color='red',   label='Disagree (1 roll)'),
        *(legend_extra or []),
    ]
    ax.legend(handles=legend, loc='upper right')
    plt.tight_layout()
    
    if not out_path:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300)
        print(f'Saved to {out_path}')
        plt.show()



def compare_piano_rolls(*piano_rolls):
    """
    Visually compare multiple piano rolls
    """
    dense = _to_dense_binary(piano_rolls)
    dense, effective_n_2d = _align_truncate(dense)
    _, agreement, majority, disagreement = _agreement_masks(dense, effective_n_2d)
    rgb = _build_rgb(dense[0].shape, agreement, majority, disagreement)
    _plot_roll(rgb, n_rolls=len(piano_rolls))


def compare_piano_rolls_labeled(piano_rolls, labels):
    """
    Visually compare up to three piano rolls with set membership labels
    """
    assert len(piano_rolls) == len(labels)

    # Build combination → (color, label) map from the provided labels
    a, b = labels[0], labels[1]
    c = labels[2] if len(labels) > 2 else None

    combination_colors = {
        (1, 0, 0): ([0.9, 0.3, 0.3], f'{a} only'),
        (0, 1, 0): ([0.3, 0.7, 0.3], f'{b} only'),
        (1, 1, 0): ([0.9, 0.8, 0.0], f'{a}+{b}'),
    }
    if c:
        combination_colors.update({
            (0, 0, 1): ([0.3, 0.3, 0.9], f'{c} only'),
            (1, 0, 1): ([0.7, 0.3, 0.9], f'{a}+{c}'),
            (0, 1, 1): ([0.2, 0.8, 0.8], f'{b}+{c}'),
            (1, 1, 1): ([1.0, 1.0, 1.0], f'{a}+{b}+{c}'),
        })

    dense = _to_dense_binary(piano_rolls)
    dense, _ = _align_truncate(dense)
    pitch_bins, n_frames = dense[0].shape

    rgb = np.zeros((pitch_bins, n_frames, 3))   # black = silent
    for combo, (color, _) in combination_colors.items():
        mask = np.ones((pitch_bins, n_frames), dtype=bool)
        for i, val in enumerate(combo[:len(dense)]):
            mask &= (dense[i] == val)
        rgb[mask] = color

    legend = [mpatches.Patch(color='black', label='Silent', linewidth=1, edgecolor='white')]
    legend += [mpatches.Patch(color=c, label=l) for _, (c, l) in combination_colors.items()]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.imshow(rgb, aspect='auto', origin='lower')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Pitch')
    ax.set_title('Piano Roll Comparison')
    ax.legend(handles=legend, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()


def _apply_majority(base_pr, all_pr_stack, agreement, majority, disagreement):
    """
    Correct a base piano roll using cross-roll consensus masks.
    
    Simple strategy: discard notes that only exist in one roll,
    fill missing notes found in majority with their mean velocity.
    """
    filtered = base_pr.copy()

    # Discard notes only one roll agrees on
    filtered[disagreement] = 0

    consensus = majority | agreement
    missing = consensus & (base_pr == 0)
    if missing.any():
        active = all_pr_stack > 0
        mean_vel = np.round(all_pr_stack.sum(axis=0) / active.sum(axis=0).clip(min=1))
        filtered[missing] = mean_vel[missing]

    return filtered


def _smoothen_pr(matrix, min_duration_threshold=50, merge_gap_threshold=30):
    """
    "Smoothen" = noise removal
    Filter and merge note events in a pitch-time matrix (88 x T).

    For each (pitch) row do:
        1. Run-length encode the row into consecutive groups of (value, start, length).
        2. Zero out any non-zero group shorter than min_duration_threshold frames (noise removal, assuming frame resolution = 1 ms).
        3. Re-merge adjacent zero groups that may have been created in step 2.
        4. Fill zero gaps shorter than merge_gap_threshold frames that are sandwiched between
        two non-zero groups, overwriting the gap with the value of the longer neighbour (all three groups then have the same velocity value = same note)
        5. Write the remaining non-zero groups into the output matrix.

    Args:
        matrix:     np.ndarray of shape (88, T), i.e. a piano roll where
                    non-zero values represent MIDI velocity
        min_duration_threshold:  Minimum length in frames a non-zero group must have to be
                    kept. Groups shorter than this are silenced, defaults to 50.
        merge_gap_threshold:  Maximum length in frames a zero gap may have to be bridged
                    between two non-zero groups. Default: 30.

    Returns:
        np.ndarray of shape (88, T), same dtype as input.
    """

    out = np.zeros_like(matrix)
    
    for i, row in enumerate(matrix):
        # 1. build RLE groups
        groups = []
        pos = 0
        for val, grp in groupby(row):
            length = sum(1 for _ in grp)
            groups.append([int(val), pos, length])
            pos += length
        
        # 2. zero out non-zero groups below threshold
        for g in groups:
            if g[0] != 0 and g[2] < min_duration_threshold:
                g[0] = 0
        
        # 3. re-merge consecutive zero groups created by step 2
        merged = []
        for g in groups:
            if merged and merged[-1][0] == g[0]:
                merged[-1][2] += g[2]
            else:
                merged.append(g[:])
        
        # 4. merge small zero gaps between two qualifying non-zero groups
        changed = True
        while changed:
            changed = False
            for j in range(1, len(merged) - 1):
                left, gap, right = merged[j-1], merged[j], merged[j+1]
                if (gap[0] == 0 and gap[2] < merge_gap_threshold
                        and left[0] != 0
                        and right[0] != 0):
                    gap[0] = left[0] if left[2] >= right[2] else right[0]
                    changed = True
        # then re-merge consecutive non-zero groups created by gap filling,
        #     using the value of the longest group
        fully_merged = []
        for g in merged:
            if fully_merged and fully_merged[-1][0] != 0 and g[0] != 0:
                # extend the existing group, keeping the value of the longer one
                prev = fully_merged[-1]
                if g[2] > prev[2]:
                    prev[0] = g[0]
                prev[2] += g[2]
            else:
                fully_merged.append(g[:])
        merged = fully_merged

        # 5. write non-zero groups
        for val, start, length in merged:
            if val != 0:
                out[i, start:start + length] = val
    
    return out


def majority_correct_piano_roll(*piano_rolls, audio_file, labels, dilation_radius=15, min_length=50, max_reonset_thresh=30, out_path=None):
    """
    "Majority-correct" multiple piano rolls into a single correct roll by measuring 
    cross-roll consistency.

    First we select the roll closest in duration to the source audio file as base,
    then we remove notes (active frames) that appear in only one roll (likely false positives),
    and fill in notes the base missed but the majority agrees on (likely omissions).
    
    To handle small timing jitter between rolls (resulting from transcription), agreement is computed on temporally dilated rolls (±dilation_radius frames), while velocity values are taken from the original undilated rolls.

    Finally, remove velocity jitter by removing short spurious notes (< min_length
    frames) and merging notes with spurious re-onsets (re-onset of same pitch within max_reonset_threshold)

    Returns a corrected version of the base roll as a csc_matrix.
    """
    assert len(piano_rolls) == len(labels)

    import librosa
    audio, sr = librosa.load(audio_file, sr=None)
    audio_dur_ms = librosa.get_duration(y=audio, sr=sr) * 1000

    dense = _to_dense_binary(piano_rolls)

    base_idx = int(np.argmin([abs(d.shape[1] - audio_dur_ms) for d in dense]))

    dense, effective_n_2d = _align_to_base(dense, base_idx)

    # Dilate for agreement computation only
    dense_dilated = _dilate_rolls(dense, radius=dilation_radius)
    _, agreement, majority, disagreement = _agreement_masks(dense_dilated, effective_n_2d)

    rgb = _build_rgb(dense[0].shape, agreement, majority, disagreement)
    _plot_roll(rgb, n_rolls=len(piano_rolls), out_path=out_path)

    # Use original (undilated) rolls for velocity computation
    base_time = dense[base_idx].shape[1]

    all_pr = []
    for pr in piano_rolls:
        d = np.array(pr.todense(), dtype=float)
        if d.shape[1] < base_time:
            d = np.pad(d, ((0, 0), (0, base_time - d.shape[1])), constant_values=0)
        else:
            d = d[:, :base_time]
        all_pr.append(d)

    all_pr_stack = np.stack(all_pr, axis=0)
    base_pr = all_pr[base_idx].copy()

    filtered_pr = _apply_majority(base_pr, all_pr_stack, agreement, majority, disagreement)
    filtered_pr = _smoothen_pr(filtered_pr, min_duration_threshold=min_length, merge_gap_threshold=max_reonset_thresh)

    return csc_matrix(filtered_pr)