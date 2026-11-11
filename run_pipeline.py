"""All steps at once.
Analysises are not part of the pipeline: run those notebooks yourself.
Without --window, step 03 only reads interim as it is. With --window it first cuts the raw exports to the window
([window] months_before), for the same participants as the reading below (new ones, the study numbers given, or --redo).

    python run_pipeline.py                        read what is new, build dataset + heat stress, reports
    python run_pipeline.py --window               first cut the raw exports of new participants to the window
    python run_pipeline.py --window A07 HR136     cut and read these participants again, then the rest as above
    python run_pipeline.py --window --redo        cut and read everyone again (e.g. after changing months_before)
    python run_pipeline.py A07 HR136              read these participants again, then the rest as above
    python run_pipeline.py --brand Fitbit Polar   only read these devices (dataset is built from all)
    python run_pipeline.py --redo                 read everyone again
    python run_pipeline.py --select hr resting_hr hr_5s    only make a selection from the saved dataset
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from heatwatch import dataset, heat_stress  # noqa: E402
from heatwatch.loading import extract, participants as pp, window  # noqa: E402
from heatwatch.reports import export, overview  # noqa: E402


def main(args):
    if args[:1] == ["--select"]:
        ds = dataset.load()
        export.save_selection(dataset.select(ds.participants, ds.trainings, ds.five_s, args[1:]), args[1:])
        return
    redo, cut = "--redo" in args, "--window" in args
    args = [a for a in args if a not in ("--redo", "--window")]
    brands = args[args.index("--brand") + 1:] if "--brand" in args else []
    ids = [a for a in args if a not in brands and a != "--brand"]

    info = pp.load()                                                              # 01
    if cut:                                                                       # 03 cut to the window
        window.cut_participants(info, study_ids=ids, brands=brands, redo=redo)
    folders = extract.read_participants(info, study_ids=ids, brands=brands, redo=redo)   # 03 read
    ds = heat_stress.add(dataset.build())                                         # 04
    check = overview.data_check(ds.info, folders if not brands else pp.find_folders(ds.info.study_id),
                                ds.participants)
    ds, check = overview.apply_exclusions(ds, check)
    export.save_dataset(ds)
    export.save_inclusion(overview.inclusion(ds.outcomes, ds.trainings))         # 05


if __name__ == "__main__":
    main(sys.argv[1:])
