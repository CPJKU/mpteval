PERF_PIANO_ROLL_PARAMS = {
    "time_unit": "sec",
    "time_div": 100,  # frames per time_unit, i.e., if time_unit is sec and time_div is 100, each frame has 10ms resolution
    "onset_only": False,
    "piano_range": True,  # 88 x num_time_steps
    "time_margin": 0,  # amount of padding before and after piano roll
    "return_idxs": False,
}

ONSET_OFFSET_TOLERANCE_NOTEWISE_EVAL = (
    5 if PERF_PIANO_ROLL_PARAMS["time_div"] == 100 else 50
)
