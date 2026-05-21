import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

import pretty_midi as pm
import mir_eval
import mir_eval.display

import warnings
warnings.filterwarnings("ignore")


def get_rgba(color, alpha=0.5):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    if not hasattr(alpha, '__iter__'):
        return rgb + (float(alpha),)
    return [rgb + (float(a),) for a in alpha]  # type: ignore[union-attr]

def plot_piano_roll_from_stream_note_arrays(stream_note_arrays, title='', ax=None, lw=0.5, alpha=0.5):
    assert len(stream_note_arrays) <= 4, 'Maximum of 4 streams are supported'

    if ax is None:
        fig = plt.figure(figsize=(4*4, 1*4))
        ax = plt.gca()
        ax.set_xlabel('Time (s)')
    else:
        fig = ax.figure

    colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:blue']

    # collect legend handles separately, don't add to axes
    legend_handles = []

    for i, (stream, notes) in enumerate(stream_note_arrays.items()):
        color = colors[i]

        legend_handles.append(mpatches.Patch(
            facecolor=get_rgba(color, alpha),  # type: ignore[arg-type]
            edgecolor='black', 
            label=stream
        ))

        for note in notes:
            ax.add_patch(Rectangle(
                (note['onset_sec'], note['pitch']),
                note['duration_sec'],
                1.0,
                facecolor=get_rgba(color, note['velocity'] / 127.),
                edgecolor='black',
                linewidth=lw,
                linestyle='-',
            ))

    ax.legend(handles=legend_handles, loc='upper left')
    ax.grid(True, which='major')
    ax.grid(True, which='minor', alpha=0.25)

    noterange = (pm.note_name_to_number('C3'), pm.note_name_to_number('C6'))
    ax.set_ybound(lower=noterange[0], upper=noterange[1])
    
    mir_eval.display.ticker_notes(ax=ax)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=12.0))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=4.0))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.set_ylabel('Note Pitch (MIDI)')
    ax.set_title(title)

    return fig, ax

def plot_correlation(ref_stream_onsets, ref_stream_feat_seq, pred_stream_feat_seq, stream='', feature='', title='', ax=None):
    if ax is None:
        fig = plt.figure(figsize=(12, 1))
        ax = plt.gca()
    else:
        fig = ax.figure
    
    corr = np.corrcoef(ref_stream_feat_seq, pred_stream_feat_seq)[0, 1]
        
    ax.plot(ref_stream_onsets, ref_stream_feat_seq, alpha=.5, linestyle='--', marker='*', label='reference')
    ax.plot(ref_stream_onsets, pred_stream_feat_seq, alpha=.5, linestyle=':', marker='*', label='prediction')
    ax.tick_params(labelbottom=True, labelleft=True)
    ax.set_ylabel(feature)
    ax.set_title(f'{title} $-$ {stream.title()} {feature.upper()} (r={corr:.2f})')
    ax.legend(loc='upper left')
    
    return fig, ax