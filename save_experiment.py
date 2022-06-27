"""
Records an experiment, i.e., a round of generating graycodes, captures and
calibration. More specifically, it saves all captures in the capture directory,
calibration_result.json and the visualizations into a single directory. All
of these data are copied instead of moved to facilitate future reuse.

We can extend this by storing more data related to the experiment inside
the directory in a systematic way.
"""

import os
import os.path
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

    # Check that all the relevant experiment information exist. These include
    # the capture and visualization directories, as well as the results file.
    assert os.path.exists("./captures"), "Top-level capture directory \
        not found."
    assert os.path.exists("./visualizations"), "Top-level visualizations \
        directory not found."
    assert os.path.exists("./calibration_result.json"), \
        "No calibration result file was found."

    # Create a new experiment directory.
    now = datetime.datetime.now()
    experiment_dir = "./experiments/E" + now.strftime("%Y%m%d_%H%M%S") \
        if name is None else f"./experiments/{name}"
    os.mkdir(experiment_dir)

    # Store the items in the experiment directory.
    shutil.copytree("./captures", f"{experiment_dir}/captures")
    shutil.copytree("./visualizations", f"{experiment_dir}/visualizations")
    shutil.copy("./calibration_result.json", experiment_dir)

if __name__ == "__main__":
    main()
