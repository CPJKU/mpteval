PERF_PIANO_ROLL_PARAMS = {
    "time_unit": "sec",
    "time_div": 100,       # frames per time_unit; if time_unit=sec and time_div=100, then fps=100 (each frame = 10ms)
    "onset_only": False,
    "piano_range": True,   # 88 x num_time_steps
    "time_margin": 0,      # amount of padding before and after piano roll
    "return_idxs": False,
}

ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL = (
    5 if PERF_PIANO_ROLL_PARAMS["time_div"] == 100 else 50
)