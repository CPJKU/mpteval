#%%

import os
import partitura as pt


PERF_PIANO_ROLL_PARAMS = {
    "time_unit": "sec",
    "time_div": 100,  # how many frames per time_unit, i.e. time_div=100 means each frame has 10ms resolution
    "onset_only": False,
    "piano_range": True,  # 88 x num_time_steps
    # "remove_silence": True, # default, we remove the silence at the start because many transcriptions start with the first note directly, to make a fair(er) comparison
    "time_margin": 0,  # amount of padding before and after piano roll
    "return_idxs": False,
}


path = '/Users/huispaty/Code/python/tri24_local/data/revnoise'
fn = '132_piece_1_reverb_3_noise_2_MiyashitaM01M_terrys_SNR12dB_T5'


perf = pt.load_performance_midi(os.path.join(path, fn + '.mid'))
pna = perf.performedparts[0].note_array()

# try:
#     tr_pr = pt.utils.compute_pianoroll(pna, **PERF_PIANO_ROLL_PARAMS)
# except ValueError:
        

# Get pitch, onset, offset from the note_info array
import numpy as np

note_info = pna
print(pna)
print(pna.shape)

pr_pitch = note_info[:, 0]
onset = note_info[:, 1]
duration = note_info[:, 2]

print(onset)

#%%
if np.any(duration < 0):
    raise ValueError("Note durations should be >= 0!")

# Get velocity if given
if note_info.shape[1] < 4:
    pr_velocity = np.ones(len(note_info))
else:
    pr_velocity = note_info[:, 3]

# Adjust pitch margin
if pitch_margin > -1:
    highest_pitch = np.max(pr_pitch)
    lowest_pitch = np.min(pr_pitch)
else:
    lowest_pitch = 0
    highest_pitch = 127

pitch_span = highest_pitch - lowest_pitch + 1

# sorted idx
idx = np.argsort(onset)
# sort notes
pr_pitch = pr_pitch[idx]
onset = onset[idx]
duration = duration[idx]
