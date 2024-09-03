import os
import csv
import pandas as pd

from eval.metrics.benchmarks import *
from eval.metrics.dynamics import *
from eval.metrics.articulation import *
from eval.metrics.timing import *

# from peamt import PEAMT # TODO peamt python 3.6 

from config import results

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm


# helpers
def remove_part_prefix_in_note_ids(score_note_array):
    """
    Remove the part prefix in note arrays created from score.xml to match them to those in a match file
    """
    # check if score note ids are correctly encoded
    if "p" in score_note_array["id"][0]:
        score_note_array["id"] = np.char.add(
            "n", np.char.partition(score_note_array["id"], sep="n")[:, -1]
        )
    return score_note_array


def get_measure_boundaries(subset, piece_path, performer):
    """
    Get measure boundaries and onset times for a given performance and piece

    Arguments:
        piece_path [str] -- path to the piece folder
        performer [str] -- performer according to the asap performance midi

    Returns:
        xml_measures [list] -- list of measure numbers according to the unfolded score xml (falls back to match measure numbers when xml != match measures)
        measure_onsets_sec [np.array] -- measure onset times in seconds
        pna [np.array] -- performance note array

        measure_onsets_sec array is also written into measure_onset_times.csv file in piece_path
    """

    # load score
    if subset == "asap_maestro":
        score_xml = os.path.join(piece_path, "xml_score.musicxml")
    elif subset == "batik_mozart":
        score_xml = os.path.join(piece_path, f"{performer}.musicxml")
    # load ground truth for that performances
    gt_match = os.path.join(piece_path, f"{performer}.match")

    # unfold alignment, score and performance for ground truth
    performance, alignment, score = pt.load_match(gt_match, create_score=True)
    pna = performance.note_array()
    pna = pna[np.argsort(pna["onset_sec"])]  # sort by onset

    # get score note array from match score
    part = score.parts[0]
    sna = part.note_array()
    sna = remove_part_prefix_in_note_ids(sna)
    # get score note array from score xml and unfold
    xml_part = pt.load_score(score_xml).parts[0]
    xml_part = pt.score.unfold_part_alignment(xml_part, alignment)
    xml_sna = xml_part.note_array()
    xml_sna = remove_part_prefix_in_note_ids(xml_sna)

    # compare xml score measures with match score measures
    xml_measures = [m for m in xml_part.iter_all(pt.score.Measure)]
    xml_measure_numbers = [m.number for m in xml_measures]
    m_measures = [m for m in part.iter_all(pt.score.Measure)]
    m_measure_numbers = [m.number for m in m_measures]

    if len(xml_measure_numbers) != len(m_measure_numbers):
        # print(f'match has {len(m_measure_numbers)} measures, score has {len(xml_measure_numbers)} measures')
        xml_measure_numbers = m_measure_numbers  # we take the match measure numbers if score and match differ

    # store measure onset times in a csv file
    onset_times_csv = os.path.join(piece_path, "measure_onset_times.csv")
    if os.path.exists(onset_times_csv):
        onset_times_df = pd.read_csv(onset_times_csv)
    else:
        onset_times_df = pd.DataFrame()
        onset_times_df["xml_m"] = xml_measure_numbers

    if f"{performer}_m" in onset_times_df.columns:
        # print(f'Found measure onset times for {performer}')
        return (
            onset_times_df[f"xml_m"].values,
            onset_times_df[f"{performer}_onset_sec"].values,
            pna,
        )
    else:
        onset_times_df[f"{performer}_m"] = m_measure_numbers

        # get the onset sec of each measure (earliest note in the measure)
        measure_onsets_sec = np.zeros(len(m_measures))
        for midx, m in enumerate(m_measures):
            # get all score notes that have their onset in the current measure
            measure_snotes = sna[
                np.where(
                    (sna["onset_div"] >= m.start.t) & (sna["onset_div"] < m.end.t)
                )[0]
            ]
            measure_snids = measure_snotes["id"]
            if measure_snids.shape[0] == 0:
                # there is no measure because there are no notes in the measure
                # print(f'Found no score note in measure {m.number} -- setting onset to NA')
                measure_onsets_sec[midx] = np.nan
                continue

            # find matched performance notes in the measure
            measure_pnids = [
                n["performance_id"]
                for n in alignment
                if n["label"] == "match" and n["score_id"] in measure_snids
            ]
            if measure_pnids == []:
                # print(f'Found no matched performance note in measure {m.number} to match with score notes {measure_snids} -- setting onset to NA')
                measure_onsets_sec[midx] = np.nan
                continue
            # get earliest onset
            measure_pnotes_mask = np.isin(pna["id"], measure_pnids)
            measure_onset_sec = np.min(pna[measure_pnotes_mask]["onset_sec"])
            measure_onsets_sec[midx] = measure_onset_sec
            # ## track measure onset snotes and pnotes
            # measure_onset_snids = measure_snotes[np.where(measure_snotes['onset_beat'] == np.min(measure_snotes['onset_beat']))]['id']
            # measure_onset_pnid = pna[np.where(
            #     pna['onset_sec'] == measure_onset_sec)[0]]['id']
            # measure_onset_snids_list.append(measure_onset_snids)
            # measure_onset_pnids_list.append(measure_onset_pnid[0])

        onset_times_df[f"{performer}_onset_sec"] = np.round(measure_onsets_sec, 4)
        onset_times_df.to_csv(onset_times_csv, index=False)

        return xml_measure_numbers, measure_onsets_sec, pna


def cut_note_array_into_segments(note_array, segment_boundaries_sec):
    """
    Convert a performance note array into a piano roll (88 x num_timesteps) and note list (list of notes as np.voids with fields (onsets, offsets, pitch, velocity)) representation, and split both according to the onset times in segment_boundaries_sec indicating measure boundaries

    Args:
        note_array [np.ndarray] -- performance note array
        segment_boundaries_sec [np.ndarray] -- measure onset times in seconds

    Returns:
        segment_notelists [list] -- list of segmented note lists
        segment_pianorolls [list] -- list of segmented piano rolls
    """
    # cut gt midi accordingly --> note list and piano roll
    # split the piano roll into segments
    indices = np.searchsorted(
        note_array["onset_sec"], segment_boundaries_sec[1:], side="left"
    )
    segment_pnas = np.split(note_array, indices)
    # create note list for each segment
    segment_notelists = [create_note_list(s) for s in segment_pnas]
    # create the piano roll
    piano_roll = pt.utils.compute_pianoroll(note_array, **PERF_PIANO_ROLL_PARAMS)
    # get the measure onset boundaries in frames (and remove silence at the beginning)
    time_div = PERF_PIANO_ROLL_PARAMS["time_div"]
    measure_onset_frames_list = np.round(
        (segment_boundaries_sec - note_array[0]["onset_sec"]) * time_div
    ).astype(int)
    # cut the piano roll
    segment_pianorolls = []
    for i, boundary in enumerate(measure_onset_frames_list):
        if i == len(measure_onset_frames_list) - 1:
            segment_pianoroll = piano_roll[:, boundary:]
        else:
            segment_pianoroll = piano_roll[
                :, boundary : measure_onset_frames_list[i + 1]
            ]
        segment_pianorolls.append(segment_pianoroll)

    return segment_notelists, segment_pianorolls


# metrics computation TODO change these functions, they're overcomplicating things
def compute_piece_benchmarks(subset, eval_piece_dir, composer, piece, split):
    """
    Compute piecewise metrics for a given piece and set of performances and their transcriptions

    Arguments:
        eval_piece_dir -- path to performance (reference) midi and transcribed (prediction) midi
        composer -- composer
        piece -- piece title
        split -- split the piece belonged to in the training of the model used for transcription

    Returns:
        writes results to piece_benchmark.csv in eval_piece_dir, with cols:
        composer, piece, split, recording, model, performer,
        frame_p, frame_r, frame_f, note_p, note_r, note_f

    """

    # get n performances
    if subset == "asap_maestro":
        performances = sorted(
            [f.split(".")[0] for f in os.listdir(eval_piece_dir) if f.endswith(".mid")]
        )
    elif subset == "batik_mozart":
        performances = [eval_piece_dir.split("/")[-1]]

    # prep csv to save results
    piece_benchmark_header = [
        "composer",
        "piece",
        "split",
        "recording",
        "model",
        "performer",
        "frame_p",
        "frame_r",
        "frame_f",
        "note_p",
        "note_r",
        "note_f",
    ]
    benchmark_results_dir = os.path.join(eval_piece_dir, "_benchmark_results")
    if not os.path.exists(benchmark_results_dir):
        os.mkdir(benchmark_results_dir)
    piece_benchmark_csv = os.path.join(benchmark_results_dir, "piece_benchmark.csv")

    with open(piece_benchmark_csv, "w+") as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        csvwriter.writerow(piece_benchmark_header)

        for pidx, performance in enumerate(performances):
            print(
                f" --- performance {pidx+1:2d}/{len(performances):2d} : {performance}"
            )
            gt_midi = os.path.join(eval_piece_dir, f"{performance}.mid")

            if subset == "asap_maestro":
                transcriptions_dirs = sorted(
                    [
                        d
                        for d in os.listdir(eval_piece_dir)
                        if os.path.isdir(os.path.join(eval_piece_dir, d))
                        and performance in d
                    ]
                )
            else:
                transcriptions_dirs = ["batik_audio", "batik_disklavier"]

            for tr_dir in transcriptions_dirs:
                print(f" ------ recording env: {tr_dir}")
                if subset == "asap_maestro":
                    recording = "disklavier" if "disklavier" in tr_dir else "maestro"
                else:
                    recording = (
                        "disklavier" if "disklavier" in tr_dir else "batik_audio"
                    )

                transcriptions = sorted(
                    [
                        f
                        for f in os.listdir(os.path.join(eval_piece_dir, tr_dir))
                        if f.endswith(".mid")
                    ]
                )
                for transcribed_midi in transcriptions:
                    print(f" --------- transcription {transcribed_midi}")
                    model = transcribed_midi.split("_")[0]
                    # piecewise metrics
                    framewise_metrics, notewise_metrics = compute_benchmarks_piecewise(
                        gt_midi,
                        os.path.join(eval_piece_dir, tr_dir, transcribed_midi),
                        verbose=False,
                    )
                    fr_p, fr_r, fr_r = framewise_metrics
                    n_p, n_r, n_f, _ = notewise_metrics

                    results = [
                        composer,
                        piece,
                        split,
                        recording,
                        model,
                        performances[pidx],
                        fr_p,
                        fr_r,
                        fr_r,
                        n_p,
                        n_r,
                        n_f,
                    ]
                    results = [
                        np.round(r, 4) if isinstance(r, float) else r for r in results
                    ]

                    csvwriter.writerow(results)

    print(f"Saved piece-level benchmark results to {piece_benchmark_csv}")
    return None


def compute_measure_benchmarks_from_segmented_note_array(
    subset,
    eval_piece_dir,
    performer,
    pna,
    transcriptions_dirs,
    measure_numbers,
    measure_onset_times_sec,
):
    """
    Compute measurewise metrics for a given piece and set of performances and their transcriptions
    """
    # specify path to save results.csv
    benchmark_results_dir = os.path.join(eval_piece_dir, "_benchmark_results")
    if not os.path.exists(benchmark_results_dir):
        os.mkdir(benchmark_results_dir)
    # init framewise dataframe to save results
    measure_metrics_framewise_csv = os.path.join(
        benchmark_results_dir, "measure_benchmark_framewise_from_pna.csv"
    )
    if os.path.exists(measure_metrics_framewise_csv):
        measure_metrics_framewise_df = pd.read_csv(measure_metrics_framewise_csv)
        # check if metrics for performer already exist
        if f"kong_{performer}_maestro_r" in measure_metrics_framewise_df.columns:
            print(f"Found measure-level benchmark metrics for {performer}")
            return None
    else:
        measure_metrics_framewise_df = pd.DataFrame()
        measure_metrics_framewise_df["xml_m"] = measure_numbers
    # init notewise dataframe to save results
    measure_metrics_notewise_csv = os.path.join(
        benchmark_results_dir, "measure_benchmark_notewise_from_pna.csv"
    )
    if os.path.exists(measure_metrics_notewise_csv):
        measure_metrics_notewise_df = pd.read_csv(measure_metrics_notewise_csv)
    else:
        measure_metrics_notewise_df = pd.DataFrame()
        measure_metrics_notewise_df["xml_m"] = measure_numbers

    # cut the reference performance note array into segments according to the onset_sec boundaries
    ref_segment_notelists, ref_segment_pianorolls = cut_note_array_into_segments(
        pna, measure_onset_times_sec
    )
    # now compute metrics for each transcription
    for tr_dir in sorted(transcriptions_dirs):
        print(f" --- evaluating recording environment {tr_dir}")
        if subset == "asap_maestro":
            recording = "disklavier" if "disklavier" in tr_dir else "maestro"
        else:
            recording = "disklavier" if "disklavier" in tr_dir else "batik_audio"

        for transcribed_midi in sorted(
            [
                f
                for f in os.listdir(os.path.join(eval_piece_dir, tr_dir))
                if f.endswith(".mid")
            ]
        ):
            print(f" ------ evaluating transcription {transcribed_midi}")
            model = transcribed_midi.split("_")[0]

            # load the transcription
            pred_midi = os.path.join(eval_piece_dir, tr_dir, transcribed_midi)
            pred_pna = (
                pt.load_performance_midi(pred_midi).performedparts[0].note_array()
            )
            # cut the transcription note array into segments according to the onset_sec boundaries
            pred_segment_notelists, pred_segment_pianorolls = (
                cut_note_array_into_segments(pred_pna, measure_onset_times_sec)
            )
            # compute framewise metrics for each segment
            segment_metrics_framewise = [
                compute_transcription_benchmark_framewise(ref_pr, pred_pr)
                for ref_pr, pred_pr in zip(
                    ref_segment_pianorolls, pred_segment_pianorolls
                )
            ]
            segment_metrics_framewise = np.array(segment_metrics_framewise)
            # track metrics
            measure_metrics_framewise_df[f"{model}_{performer}_{recording}_r"] = (
                segment_metrics_framewise[:, 0]
            )
            measure_metrics_framewise_df[f"{model}_{performer}_{recording}_p"] = (
                segment_metrics_framewise[:, 1]
            )
            measure_metrics_framewise_df[f"{model}_{performer}_{recording}_f1"] = (
                segment_metrics_framewise[:, 2]
            )

            # compute notewise metrics for each segment
            segment_metrics_notewise = [
                compute_transcription_benchmark_notewise(ref_nl, pred_nl)
                for ref_nl, pred_nl in zip(
                    ref_segment_notelists, pred_segment_notelists
                )
            ]
            segment_metrics_notewise = np.array(segment_metrics_notewise)
            # track metrics
            measure_metrics_notewise_df[f"{model}_{performer}_{recording}_r"] = (
                segment_metrics_notewise[:, 0]
            )
            measure_metrics_notewise_df[f"{model}_{performer}_{recording}_p"] = (
                segment_metrics_notewise[:, 1]
            )
            measure_metrics_notewise_df[f"{model}_{performer}_{recording}_f1"] = (
                segment_metrics_notewise[:, 2]
            )

    # save results
    measure_metrics_framewise_df.to_csv(measure_metrics_framewise_csv, index=False)
    measure_metrics_notewise_df.to_csv(measure_metrics_notewise_csv, index=False)

    print(f"Saved measure-level benchmark metrics to {eval_piece_dir}")
    return None


def compute_measure_benchmarks_from_segmented_performed_parts(subset, piece_path):
    """
    Compute measure benchmarks from the segmented performed parts, write results to csv with columns:

    mn | recording | model | performer | frame_p | frame_r | frame_f | note_p | note_r | note_f
          maestro    kong
         disklavier  oaf

    Arguments:
        subset -- _description_
        piece_path -- _description_
        performer -- _description_
        transcriptions_dirs -- _description_
        measure_numbers -- _description_
        measure_onset_times_sec -- _description_

    Returns:
        _description_
    """

    # get n performances
    if subset == "asap_maestro":
        performances = sorted(
            [f.split(".")[0] for f in os.listdir(piece_path) if f.endswith(".mid")]
        )
        original_audio_measure_segment_midis_dir = os.path.join(
            piece_path, "_measure_segments_maestro"
        )
        disklavier_measure_segment_midis_dir = os.path.join(
            piece_path, "_measure_segments_disklavier"
        )
        record_envs = ["maestro", "disklavier"]

    elif subset == "batik_mozart":
        performances = [piece_path.split("/")[-1]]
        original_audio_measure_segment_midis_dir = os.path.join(
            piece_path, "_measure_segments_batik_audio"
        )
        disklavier_measure_segment_midis_dir = os.path.join(
            piece_path, "_measure_segments_disklavier"
        )
        record_envs = ["batik_audio", "disklavier"]

    # prep csv to save results
    measure_benchmark_header = [
        "mn",
        "model",
        "performer",
        "frame_p",
        "frame_r",
        "frame_f",
        "note_p",
        "note_r",
        "note_f",
    ]

    for record_env, segment_midis_dir in zip(
        record_envs,
        [
            original_audio_measure_segment_midis_dir,
            disklavier_measure_segment_midis_dir,
        ],
    ):
        print(f" --- evaluating recording environment {record_env}")
        benchmark_results_dir = os.path.join(piece_path, "_benchmark_results")
        if not os.path.exists(benchmark_results_dir):
            os.mkdir(benchmark_results_dir)
        measure_benchmark_csv = os.path.join(
            benchmark_results_dir, f"measure_benchmark__{record_env}.csv"
        )

        with open(measure_benchmark_csv, "w+") as f:
            # creating a csv writer object
            csvwriter = csv.writer(f)
            csvwriter.writerow(measure_benchmark_header)

            for pidx, performance in enumerate(performances):
                print(
                    f" ------ performance {pidx+1:2d}/{len(performances):2d} : {performance}"
                )

                for m_dir in sorted(os.listdir(segment_midis_dir)):
                    mn = m_dir.split("_")[-1]
                    print(f" ------ measure {mn}")
                    # get the gt midi
                    gt_midi = os.path.join(
                        segment_midis_dir, m_dir, f"{performance}.mid"
                    )
                    if not os.path.exists(gt_midi):
                        continue
                    else:
                        # get the transcriptions
                        transcriptions = sorted(
                            [
                                f
                                for f in os.listdir(
                                    os.path.join(segment_midis_dir, m_dir)
                                )
                                if f != f"{performance}.mid" and performance in f
                            ]
                        )
                        for transcription in transcriptions:
                            print(f" --------- transcription {transcription}")
                            model = transcription.split(".")[0].split("_")[-1]

                            framewise_metrics, notewise_metrics = (
                                compute_benchmarks_piecewise(
                                    gt_midi,
                                    os.path.join(
                                        segment_midis_dir, m_dir, transcription
                                    ),
                                    verbose=False,
                                )
                            )
                            fr_p, fr_r, fr_r = framewise_metrics
                            n_p, n_r, n_f, _ = notewise_metrics

                            results = [
                                mn,
                                record_env,
                                model,
                                performance,
                                fr_p,
                                fr_r,
                                fr_r,
                                n_p,
                                n_r,
                                n_f,
                            ]
                            results = [
                                np.round(r, 4) if isinstance(r, float) else r
                                for r in results
                            ]

                            csvwriter.writerow(results)

    return None


####################################



def compute_piece_frame(subset, piece_path, composer, title, split):
    """
    """

    files = sorted(os.listdir(piece_path))

    performers = [f.split('.')[0] for f in files if f.endswith('.match')]

    dyn_header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording',
                  'p_f', 'r_f', 'f_f']

    dyn_results_csv = os.path.join(results, 'IR_frame.csv')
    if os.path.exists(dyn_results_csv):
        mode = 'a'
    else:
        mode = 'w+'

    with open(dyn_results_csv, mode) as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        if mode == 'w+':
            csvwriter.writerow(dyn_header)

        for pidx, performer in enumerate(performers):
            gt_midi = os.path.join(piece_path, f'{performer}.mid')
            gt_perf = pt.load_performance_midi(gt_midi)
            gt_na = gt_perf.note_array()
            gt_pr = pt.utils.compute_pianoroll(gt_na, **PERF_PIANO_ROLL_PARAMS)

            print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            if subset == 'asap_maestro':
                oa_transcriptions = os.path.join(
                    piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(
                    piece_path, f'{performer}_disklavier')

            else:
                oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                disklavier_transcriptions = os.path.join(
                    piece_path, 'batik_disklavier')

            print(f' --- maestro transcriptions:')
            for maestro_tr in os.listdir(oa_transcriptions):
                if not maestro_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {maestro_tr}')
                    tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    tr_na = tr_perf.note_array()
                    tr_pr = pt.utils.compute_pianoroll(tr_na, **PERF_PIANO_ROLL_PARAMS)
                    p, r, f = compute_transcription_benchmark_framewise(
                        gt_pr, tr_pr)

                    model = maestro_tr.split('.')[0].split('_')[0]

                    if subset == 'asap_maestro':
                        recording = 'maestro'
                    else:
                        recording = 'batik'

                    res = [composer, title, split, model,
                           performer, model, recording, p, r, f]
                    csvwriter.writerow(res)

            print(f' --- disklavier transcriptions:')
            for disklavier_tr in os.listdir(disklavier_transcriptions):
                if not disklavier_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {disklavier_tr}')
                    tr_midi = os.path.join(
                        disklavier_transcriptions, disklavier_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    tr_na = tr_perf.note_array()
                    tr_pr = pt.utils.compute_pianoroll(tr_na, **PERF_PIANO_ROLL_PARAMS)
                    p, r, f = compute_transcription_benchmark_framewise(
                        gt_pr, tr_pr)

                    model = disklavier_tr.split('.')[0].split('_')[0]

                    res = [composer, title, split, performer,
                           model, 'disklavier', p, r, f]
                    csvwriter.writerow(res)
    return None

def compute_piece_note_with_offset(subset, piece_path, composer, title, split):
    """
    """

    files = sorted(os.listdir(piece_path))

    performers = [f.split('.')[0] for f in files if f.endswith('.match')]

    dyn_header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording',
                  'p_no', 'r_no', 'f_no']

    dyn_results_csv = os.path.join(results, 'IR_note_offset.csv')
    if os.path.exists(dyn_results_csv):
        mode = 'a'
    else:
        mode = 'w+'

    with open(dyn_results_csv, mode) as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        if mode == 'w+':
            csvwriter.writerow(dyn_header)

        for pidx, performer in enumerate(performers):
            gt_midi = os.path.join(piece_path, f'{performer}.mid')
            gt_perf = pt.load_performance_midi(gt_midi)
            gt_na = gt_perf.note_array()
            gt_notelist = create_note_list(gt_na)

            print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            if subset == 'asap_maestro':
                oa_transcriptions = os.path.join(
                    piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(
                    piece_path, f'{performer}_disklavier')

            else:
                oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                disklavier_transcriptions = os.path.join(
                    piece_path, 'batik_disklavier')

            print(f' --- maestro transcriptions:')
            for maestro_tr in os.listdir(oa_transcriptions):
                if not maestro_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {maestro_tr}')
                    tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    tr_na = tr_perf.note_array()
                    tr_notelist = create_note_list(tr_na)
                    p, r, f, _ = compute_transcription_benchmark_notewise(
                        gt_notelist, tr_notelist)

                    model = maestro_tr.split('.')[0].split('_')[0]

                    if subset == 'asap_maestro':
                        recording = 'maestro'
                    else:
                        recording = 'batik'

                    res = [composer, title, split, model,
                           performer, model, recording, p, r, f]
                    csvwriter.writerow(res)

            print(f' --- disklavier transcriptions:')
            for disklavier_tr in os.listdir(disklavier_transcriptions):
                if not disklavier_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {disklavier_tr}')
                    tr_midi = os.path.join(
                        disklavier_transcriptions, disklavier_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    tr_na = tr_perf.note_array()
                    tr_notelist = create_note_list(tr_na)
                    p, r, f, _ = compute_transcription_benchmark_notewise(
                        gt_notelist, tr_notelist)

                    model = disklavier_tr.split('.')[0].split('_')[0]

                    res = [composer, title, split, performer,
                           model, 'disklavier', p, r, f]
                    csvwriter.writerow(res)
    return None

def compute_piece_note_with_offset_velocity(subset, piece_path, composer, title, split):
    """
    """

    files = sorted(os.listdir(piece_path))

    performers = [f.split('.')[0] for f in files if f.endswith('.match')]

    dyn_header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording', 
                  'p_nov', 'r_nov', 'f_nov']

    dyn_results_csv = os.path.join(results, 'IR_note_offset_velocity.csv')
    if os.path.exists(dyn_results_csv):
        mode = 'a'
    else:
        mode = 'w+'

    with open(dyn_results_csv, mode) as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        if mode == 'w+':
            csvwriter.writerow(dyn_header)

        for pidx, performer in enumerate(performers):
            gt_midi = os.path.join(piece_path, f'{performer}.mid')
            gt_perf = pt.load_performance_midi(gt_midi)
            gt_na = gt_perf.note_array()
            gt_notelist = create_note_list(gt_na)

            print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            if subset == 'asap_maestro':
                oa_transcriptions = os.path.join(
                    piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(
                    piece_path, f'{performer}_disklavier')

            else:
                oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                disklavier_transcriptions = os.path.join(
                    piece_path, 'batik_disklavier')

            print(f' --- maestro transcriptions:')
            for maestro_tr in os.listdir(oa_transcriptions):
                if not maestro_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {maestro_tr}')
                    tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    tr_na = tr_perf.note_array()
                    tr_notelist = create_note_list(tr_na)
                    p, r, f, _ = compute_transcription_benchmark_notewise_with_velocity(gt_notelist, tr_notelist)

                    model = maestro_tr.split('.')[0].split('_')[0]

                    if subset == 'asap_maestro':
                        recording = 'maestro'
                    else:
                        recording = 'batik'

                    res = [composer, title, split, model,
                           performer, model, recording, p, r, f]
                    csvwriter.writerow(res)

            print(f' --- disklavier transcriptions:')
            for disklavier_tr in os.listdir(disklavier_transcriptions):
                if not disklavier_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {disklavier_tr}')
                    tr_midi = os.path.join(
                        disklavier_transcriptions, disklavier_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    tr_na = tr_perf.note_array()
                    tr_notelist = create_note_list(tr_na)
                    p, r, f, _ = compute_transcription_benchmark_notewise_with_velocity(gt_notelist, tr_notelist)

                    model = disklavier_tr.split('.')[0].split('_')[0]

                    res = [composer, title, split, performer,
                           model, 'disklavier', p, r, f]
                    csvwriter.writerow(res)
    return None


def compute_piece_dynamics_metrics(subset, piece_path, composer, title, split):
    """
    """    
    
    files = sorted(os.listdir(piece_path)) 
    
    performers = [f.split('.')[0] for f in files if f.endswith('.match')]
    
    dyn_header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording', 'dyn_corr']
    
    dyn_results_csv = os.path.join(results, 'musical_dynamics.csv')
    if os.path.exists(dyn_results_csv):
        mode = 'a'
    else: 
        mode = 'w+'
    
    with open(dyn_results_csv, mode) as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        if mode == 'w+' : csvwriter.writerow(dyn_header)
        
        for pidx, performer in enumerate(performers):
            gt_midi = os.path.join(piece_path, f'{performer}.mid')
            gt_perf = pt.load_performance_midi(gt_midi)
            
            print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            if subset == 'asap_maestro':
                oa_transcriptions = os.path.join(piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(piece_path, f'{performer}_disklavier')
                
            else:
                oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                disklavier_transcriptions = os.path.join(piece_path, 'batik_disklavier')
            
            print(f' --- maestro transcriptions:')
            for maestro_tr in os.listdir(oa_transcriptions):
                if not maestro_tr.endswith('.mid'): continue
                else:
                    print(f' ------ transcription {maestro_tr}')
                    tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    dyn_corr = np.round(dynamics_metrics_from_perf(tr_perf, gt_perf), 4)
                    
                    model = maestro_tr.split('.')[0].split('_')[0]
                    
                    if subset == 'asap_maestro':
                        recording = 'maestro'
                    else: recording = 'batik'
                    
                    res = [composer, title, split, model,
                           performer, model, recording, dyn_corr]
                    csvwriter.writerow(res)
                    
            print(f' --- disklavier transcriptions:')
            for disklavier_tr in os.listdir(disklavier_transcriptions):
                if not disklavier_tr.endswith('.mid'): continue
                else:
                    print(f' ------ transcription {disklavier_tr}')
                    tr_midi = os.path.join(disklavier_transcriptions, disklavier_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    dyn_corr = np.round(dynamics_metrics_from_perf(tr_perf, gt_perf), 4)
                    
                    model = disklavier_tr.split('.')[0].split('_')[0]
                    
                    res = [composer, title, split, performer, model, 'disklavier', dyn_corr]
                    csvwriter.writerow(res)
    return None


def compute_piece_articulation_metrics(subset, piece_path, composer, title, split):
    """
    """

    files = sorted(os.listdir(piece_path))

    performers = [f.split('.')[0] for f in files if f.endswith('.match')]
    
    
    art_header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording', 
                  'melody_kor_corr_64', 'bass_kor_corr_64', 'ratio_kor_corr_64', 
                  'melody_kor_corr_127', 'bass_kor_corr_127', 'ratio_kor_corr_127'
                  ]

    art_results_csv = os.path.join(results, f'musical_articulation.csv')
    if os.path.exists(art_results_csv):
        mode = 'a'
    else:
        mode = 'w+'

    with open(art_results_csv, mode) as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        if mode == 'w+':
            csvwriter.writerow(art_header)

        for pidx, performer in enumerate(performers):
            gt_midi = os.path.join(piece_path, f'{performer}.mid')
            gt_perf = pt.load_performance_midi(gt_midi)

            print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            if subset == 'asap_maestro':
                oa_transcriptions = os.path.join(
                    piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(
                    piece_path, f'{performer}_disklavier')

            else:
                oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                disklavier_transcriptions = os.path.join(
                    piece_path, 'batik_disklavier')

            print(f' --- maestro transcriptions:')
            for maestro_tr in os.listdir(oa_transcriptions):
                if not maestro_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {maestro_tr}')
                    tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    
                    art_metrics = articulation_metrics_from_perf(tr_perf, gt_perf)
                    art_metrics_64, art_metrics_127 = art_metrics[0], art_metrics[1]
                    
                    melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, _ = art_metrics_64
                    melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127, _ = art_metrics_127    

                    model = maestro_tr.split('.')[0].split('_')[0]

                    if subset == 'asap_maestro':
                        recording = 'maestro'
                    else:
                        recording = 'batik'

                    res = [composer, title, split, model,
                           performer, model, recording, 
                           melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127]
                    res = [np.round(r, 4) if isinstance(r, float) else r for r in res]
                    csvwriter.writerow(res)
                                    
            print(f' --- disklavier transcriptions:')
            for disklavier_tr in os.listdir(disklavier_transcriptions):
                if not disklavier_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {disklavier_tr}')
                    tr_midi = os.path.join(
                        disklavier_transcriptions, disklavier_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)
                    
                    art_metrics = articulation_metrics_from_perf(tr_perf, gt_perf)
                    art_metrics_64, art_metrics_127 = art_metrics[0], art_metrics[1]

                    melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, _ = art_metrics_64
                    melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127, _ = art_metrics_127

                    model = disklavier_tr.split('.')[0].split('_')[0]

                    res = [composer, title, split, 
                           performer, model, 'disklavier', 
                           melody_kor_corr_64, bass_kor_corr_64, ratio_kor_corr_64, melody_kor_corr_127, bass_kor_corr_127, ratio_kor_corr_127]
                    res = [np.round(r, 4) if isinstance(
                        r, float) else r for r in res]
                    
                    csvwriter.writerow(res)
                        
    return None


def compute_piece_timing_metrics(subset, piece_path, composer, title, split):
    """
    """

    files = sorted(os.listdir(piece_path))

    performers = [f.split('.')[0] for f in files if f.endswith('.match')]

    timing_header = ['composer', 'title', 'split',
                  'performer', 'model', 'recording',
                  'melody_ioi_corr', 'acc_ioi_corr', 'ratio_ioi_corr',
                  'melody_ioi_dtw_dist', 'acc_ioi_dtw_dist', 
                  'melody_ioi_hist_kld', 'acc_ioi_hist_kld'
                  ]

    timing_results_csv = os.path.join(results, 'musical_timing.csv')
    if os.path.exists(timing_results_csv):
        mode = 'a'
    else:
        mode = 'w+'

    with open(timing_results_csv, mode) as f:
        # creating a csv writer object
        csvwriter = csv.writer(f)
        if mode == 'w+':
            csvwriter.writerow(timing_header)

        for pidx, performer in enumerate(performers):
            gt_midi = os.path.join(piece_path, f'{performer}.mid')
            gt_perf = pt.load_performance_midi(gt_midi)

            print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
            if subset == 'asap_maestro':
                oa_transcriptions = os.path.join(
                    piece_path, f'{performer}_maestro')
                disklavier_transcriptions = os.path.join(
                    piece_path, f'{performer}_disklavier')

            else:
                oa_transcriptions = os.path.join(piece_path, 'batik_audio')
                disklavier_transcriptions = os.path.join(
                    piece_path, 'batik_disklavier')

            print(f' --- maestro transcriptions:')
            for maestro_tr in os.listdir(oa_transcriptions):
                if not maestro_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {maestro_tr}')
                    tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)

                    timing_metrics = timing_metrics_from_perf(
                        tr_perf, gt_perf)
                
                    model = maestro_tr.split('.')[0].split('_')[0]

                    if subset == 'asap_maestro':
                        recording = 'maestro'
                    else:
                        recording = 'batik'

                    res = [composer, title, split, model,
                           performer, model, recording,
                           *timing_metrics[0]]
                    res = [np.round(r, 4) if isinstance(
                        r, float) else r for r in res]
                    csvwriter.writerow(res)

            print(f' --- disklavier transcriptions:')
            for disklavier_tr in os.listdir(disklavier_transcriptions):
                if not disklavier_tr.endswith('.mid'):
                    continue
                else:
                    print(f' ------ transcription {disklavier_tr}')
                    tr_midi = os.path.join(
                        disklavier_transcriptions, disklavier_tr)
                    tr_perf = pt.load_performance_midi(tr_midi)

                    timing_metrics = timing_metrics_from_perf(
                        tr_perf, gt_perf)

                    model = disklavier_tr.split('.')[0].split('_')[0]

                    res = [composer, title, split,
                           performer, model, 'disklavier',
                           *timing_metrics[0]]
                    res = [np.round(r, 4) if isinstance(
                        r, float) else r for r in res]

                    csvwriter.writerow(res)

    return None

# def compute_piece_peamt(subset, piece_path, composer, title, split):
#     """
#     """

#     files = sorted(os.listdir(piece_path))

#     performers = [f.split('.')[0] for f in files if f.endswith('.match')]

#     timing_header = ['composer', 'title', 'split',
#                   'performer', 'model', 'recording',
#                   'peamt'
#                   ]

#     timing_results_csv = os.path.join(results, 'perceptual_peamt.csv')
#     if os.path.exists(timing_results_csv):
#         mode = 'a'
#     else:
#         mode = 'w+'

#     with open(timing_results_csv, mode) as f:
#         # creating a csv writer object
#         csvwriter = csv.writer(f)
#         if mode == 'w+':
#             csvwriter.writerow(timing_header)

#         for pidx, performer in enumerate(performers):
#             gt_midi = os.path.join(piece_path, f'{performer}.mid')
#             print(f' --- performer {pidx+1}/{len(performers):2d}: {performer}')
#             if subset == 'asap_maestro':
#                 oa_transcriptions = os.path.join(
#                     piece_path, f'{performer}_maestro')
#                 disklavier_transcriptions = os.path.join(
#                     piece_path, f'{performer}_disklavier')

#             else:
#                 oa_transcriptions = os.path.join(piece_path, 'batik_audio')
#                 disklavier_transcriptions = os.path.join(
#                     piece_path, 'batik_disklavier')

#             print(f' --- maestro transcriptions:')
#             for maestro_tr in os.listdir(oa_transcriptions):
#                 if not maestro_tr.endswith('.mid'):
#                     continue
#                 else:
#                     print(f' ------ transcription {maestro_tr}')
#                     tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    
#                     eval = PEAMT()
#                     peamt = eval.evaluate_from_midi(gt_midi,tr_midi)
                
#                     model = maestro_tr.split('.')[0].split('_')[0]

#                     if subset == 'asap_maestro':
#                         recording = 'maestro'
#                     else:
#                         recording = 'batik'

#                     res = [composer, title, split, model,
#                            performer, model, recording, res]
#                     res = [np.round(r, 4) if isinstance(
#                         r, float) else r for r in res]
#                     csvwriter.writerow(res)

#             print(f' --- disklavier transcriptions:')
#             for disklavier_tr in os.listdir(disklavier_transcriptions):
#                 if not disklavier_tr.endswith('.mid'):
#                     continue
#                 else:
#                     print(f' ------ transcription {disklavier_tr}')
#                     tr_midi = os.path.join(
#                         disklavier_transcriptions, disklavier_tr)
#                     tr_midi = os.path.join(oa_transcriptions, maestro_tr)
                    
#                     eval = PEAMT()
#                     peamt = eval.evaluate_from_midi(gt_midi,tr_midi)

#                     model = disklavier_tr.split('.')[0].split('_')[0]

#                     res = [composer, title, split,
#                            performer, model, 'disklavier',
#                            peamt]
#                     res = [np.round(r, 4) if isinstance(
#                         r, float) else r for r in res]

#                     csvwriter.writerow(res)

#     return None

####################################
# compute metrics for a subset of the eval data
####################################

def compute_subset_metrics(
    subset, subset_meta_csv, eval_set_path, metric_type="piece_benchmark"
):
    """
    Compute metrics for a given subset of the eval data
    Args:
        subset [str] -- name of the subset
        subset_meta_csv [str] -- path to the meta csv for the subset
        eval_set_path [str] -- path to the eval data
        metric_type [str] -- type of metric to compute
    """
    # read the meta csv
    if not isinstance(subset_meta_csv, pd.DataFrame):
        subset_meta_csv = pd.read_csv(subset_meta_csv)

    if metric_type == "piece_benchmark":

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(
                    f"Computing benchmarks for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances"
                )
                compute_piece_benchmarks(subset, piece_path, composer, title, split)

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing benchmarks for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_benchmarks(
                    subset, piece_path, "Mozart", row["title"], None
                )

    elif metric_type == "measure_benchmark":

        for i, row in tqdm(subset_meta_csv.iterrows()):

            if subset == "asap_maestro":
                print(
                    f'\nComputing measure-level benchmarks  in {row["split"]}/{row["folder"]} for {row["midi_performance"]}'
                )
                piece_path = os.path.join(eval_set_path, row["split"], row["folder"])
                performer = row["midi_performance"].split("/")[-1].split(".mid")[0]
                transcriptions_dirs = sorted(
                    [
                        d
                        for d in os.listdir(piece_path)
                        if os.path.isdir(os.path.join(piece_path, d)) and performer in d
                    ]
                )

            elif subset == "batik_mozart":
                print(f'\nComputing measure-level benchmarks  for {row["title"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                performer = row["title"]
                transcriptions_dirs = ["batik_audio", "batik_disklavier"]

            # get the measure numbers, their onset times and the performance note array
            measure_numbers, measure_onset_times_sec, pna = get_measure_boundaries(
                subset, piece_path, performer
            )

            compute_measure_benchmarks_from_segmented_note_array(
                subset,
                piece_path,
                performer,
                pna,
                transcriptions_dirs,
                measure_numbers,
                measure_onset_times_sec,
            )

    # TMP for testing: compute it once from segmented midis, choose one
    elif metric_type == "measure_benchmark_segmented_midis":
        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                compute_measure_benchmarks_from_segmented_performed_parts(
                    subset, piece_path
                )

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_measure_benchmarks_from_segmented_performed_parts(
                    subset, piece_path
                )

    # TODOTODOTODO change the functions above

    elif metric_type == "frame":

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")

                compute_piece_frame(
                    subset, piece_path, composer, title, split)

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_frame(
                    subset, piece_path, "Mozart", row["title"], "Batik"
                )

    elif metric_type == "note_offset":

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(
                    f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")

                compute_piece_note_with_offset(
                    subset, piece_path, composer, title, split)

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_note_with_offset(
                    subset, piece_path, "Mozart", row["title"], "Batik"
                )
    
    elif metric_type == "note_offset_velocity":

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(
                    f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")

                compute_piece_note_with_offset_velocity(
                    subset, piece_path, composer, title, split)

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_note_with_offset_velocity(
                    subset, piece_path, "Mozart", row["title"], "Batik"
                )

    elif metric_type == "dynamics":

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")
    
                compute_piece_dynamics_metrics(
                    subset, piece_path, composer, title, split)
                
        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_dynamics_metrics(
                    subset, piece_path, "Mozart", row["title"], "Batik"
                )
    
    elif metric_type == 'articulation':

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(
                    f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")

                compute_piece_articulation_metrics(
                    subset, piece_path, composer, title, split)

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_articulation_metrics(
                    subset, piece_path, "Mozart", row["title"], "Batik"
                )

    elif metric_type == 'timing':

        if subset == "asap_maestro":
            grouping_var = ["split", "folder"]

            for _, piece_data in tqdm(subset_meta_csv.groupby(grouping_var)):
                piece_path = os.path.join(
                    eval_set_path,
                    piece_data["split"].unique()[0],
                    piece_data["folder"].unique()[0],
                )
                composer = piece_data["composer"].unique()[0]
                title = piece_data["title"].unique()[0]
                split = piece_data["split"].unique()[0]

                print(69 * "-")
                print(
                    f"Computing {metric_type} metrics for {split}/{title} by {composer} for {piece_data['midi_performance'].nunique()} performances")

                compute_piece_timing_metrics(
                    subset, piece_path, composer, title, split)

        elif subset == "batik_mozart":
            for _, row in tqdm(subset_meta_csv.iterrows()):
                print(69 * "-")
                print(f'Computing {metric_type} for {row["midi_performance"]}')
                piece_path = os.path.join(eval_set_path, row["title"])
                compute_piece_timing_metrics(
                    subset, piece_path, "Mozart", row["title"], "Batik"
                )

    else:
        raise NotImplementedError(f"Metric type {metric_type} not supported")

    return None
