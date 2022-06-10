#!/usr/bin/env python3

import datetime
import argparse
import os.path
import shutil


def main():
    # Accept an optional argument describing the name of the experiment.
    parser = argparse.ArgumentParser(
        description="Save data relating to the experiment into a directory",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-name", type=str, default=None)
    args = parser.parse_args()
    name = args.name

    now = datetime.datetime.now()
    results_filename = f"E{now.strftime('%Y%m%d_%H%M%S')}" if name is None else name
    results_path = f"./results/{results_filename}.json"
    
    assert os.path.exists("./calibration_result.json")
    shutil.copyfile("./calibration_result.json", results_path)


if __name__ == "__main__":
    main()
