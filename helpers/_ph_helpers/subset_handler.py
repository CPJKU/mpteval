# %%
import pandas as pd

import sys
sys.path.append('..')

from config import EVAL_DATA_PATHS

EVAL_SET_META_COLS = [
    "composer",
    "title",
    "split",
    "version",
    "folder",
    "xml_score",
    "midi_score",
    "midi_performance",
    "audio_performance",
    "midi_path",
    "audio_path",
    "duration_sec",
    "robust_note_alignment",
]

REVNOISE_META_COLS = [
    "subset",
    "composer",
    "title",
    "split",
    "folder",
    "audio_env",
    "piece_id"
    ]

reverb_irf_levels = {0: 'norev', 1: 's', 2: 'm', 3: 'l'}
snr_levels = {0: 'nonoise', 1: 24, 2: 12, 3: 6}  # [dB]

'''
REVERB IR FILTERS
1) Short reverberation time: Genesis 6 Studio – Live Room Drum Set Up RT60@1kHz: 0.19s
2) Medium reverberation time: Jack Lyons Concert Hall
3) Long reverberation time: Terry’s Factory Warehouse

NOISE : adding white noise at different SNR levels
snr_levels = {0: 'nonoise', 1: 24, 2: 12, 3: 6}  # [dB]

'''

REVNOISE_ID_PIECE_MAPPING = {
    1: {
        'subset': 'asap_maestro',
        'composer': 'Bach',
        'title': 'Fugue_bwv_848',
        'split': 'train',
        'performer': 'MiyashitaM01M',
        'audio_env': 'disklavier',
    },
    2: {
        'subset': 'asap_maestro',
        'composer': 'Beethoven',
        'title': 'Piano_Sonatas_21-1_no_repeat',
        'split': 'train',
        'performer': 'Sladek02M',
        'audio_env': 'maestro',
    },
    3: {
        'subset': 'asap_maestro',
        'composer': 'Chopin',
        'title': 'Scherzos_31',
        'split': 'train',
        'performer': 'Jussow11M',
        'audio_env': 'disklavier',
    },
    4: {
        'subset': 'asap_maestro',
        'composer': 'Liszt',
        'title': 'Transcendental_Etudes_11',
        'split': 'test',
        'performer': 'Huang18M',
        'audio_env': 'maestro',
    },
    5: {
        'subset': 'asap_maestro',
        'composer': 'Rachmaninoff',
        'title': 'Preludes_op_23_4',
        'split': 'train',
        'performer': 'WuuE07M',
        'audio_env': 'disklavier',
    },
    6: {
        'subset': 'asap_maestro',
        'composer': 'Schubert',
        'title': 'Piano_Sonatas_664-1',
        'split': 'test',
        'performer': 'KabuliL10M',
        'audio_env': 'disklavier',
    },
    7: {
        'subset': 'batik_mozart',
        'composer': 'Mozart',
        'title': 'kv332_1',
        'split': 'batik_mozart',
        'performer': 'batik',
        'audio_env': 'batik_audio'
    }    
}

meta = pd.DataFrame(REVNOISE_ID_PIECE_MAPPING).T
meta['piece_id'] = meta.index

revnoise_data_path = EVAL_DATA_PATHS['revnoise']['data_path']
revnoise_meta = EVAL_DATA_PATHS['revnoise']['meta']

meta.to_csv(revnoise_meta, index=False)