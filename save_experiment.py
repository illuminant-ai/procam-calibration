"""
Records an experiment, i.e., a round of generating graycodes, captures and
calibration. More specifically, it saves all capture_* directories,
calibration_result.json and the visualization files -- viz*.png -- into a
single directory.

The capture directories are copied instead of moved to facilitate further
experiments with the same capture. The calibration_result.json file is
also copied to keep the most recent results for inspection purposes. The
remaining visualization files are simply moved into the experiment folder.

We can extend this by storing more data related to the experiment inside
the directory in a systematic way.
"""

import os
import os.path
import glob
import datetime
import shutil
import argparse


def main():
    # Accept an optional argument describing the name of the experiment.
    parser = argparse.ArgumentParser(
        description="Save data relating to the experiment into a directory",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-name", type=str, default=None)
    args = parser.parse_args()
    name = args.name

    # Find all the capture directories.
    captures = sorted(glob.glob(".\\capture_*"))
    assert len(captures) > 0, "No capture directories were found."

    # Find all the visualization directories.
    visualizations = glob.glob(".\\viz*.png")
    assert len(visualizations) > 0, "No visualizations were found."

    # Check that the calibration_result.json file exists.
    assert os.path.exists(".\\calibration_result.json"), \
        "No calibration result file was found."

    # Create the new experiment directory.
    now = datetime.datetime.now()
    experiment_dir = ".\\experiments\\e_" + now.strftime("%Y%m%d_%H%M%S") \
        if name is None else f".\\experiments\\{name}"
    os.mkdir(experiment_dir)

    # Store the items in the experiment directory.
    for capture_dir in captures:
        n = int(capture_dir.split("_")[1])
        dir = f"{experiment_dir}\\capture_{n}"
        shutil.copytree(capture_dir, dir, dirs_exist_ok=True)
    for vis_file in visualizations:
        shutil.move(vis_file, experiment_dir)
    shutil.copy(".\\calibration_result.json", experiment_dir)

if __name__ == "__main__":
    main()
