"""
preprocessor.py

This module will apply thresholds to datafiles depending on the source
"""
import os

import prescient.gosm.gosm_options as gosm_options
from prescient.gosm.sources import Source, source_from_csv


def parse_data_files(filename):
    """
    This file reads through a file specifying the files to preprocess and reads
    in the files creating Source objects for each of them and then returns
    a list of the Sources.

    The file should be in the following format:
    file1,type1
    file2,type2
    ...

    Any comments beginning with # will be ignored

    Args:
        filename: The name of the file with the list of files
    Returns:
        List[Source]: A list of Source objects read from the files
    """
    list_of_sources = []
    with open(filename) as f:
        for line in f:
            # Eat up the comment
            text, *_ = line.split('#')
            text = text.strip()
            if text:
                filename, source_type = text.split(',')
                *_, name = filename.split(os.sep)
                source = source_from_csv(filename, name, source_type)
                list_of_sources.append(source)
    return list_of_sources

def main():
    # Set the options.
    gosm_options.set_globals()

    # Create output directory.
    if not (os.path.isdir(gosm_options.output_directory)):
        os.mkdir(gosm_options.output_directory)

    sources = parse_data_files(gosm_options.preprocessor_list)

    print("Thresholding...")
    for source in sources:
        if source.source_type == 'wind':
            lower = gosm_options.wind_power_neg_threshold
            upper = gosm_options.wind_power_pos_threshold
        elif source.source_type == 'solar':
            lower = gosm_options.solar_power_neg_threshold
            upper = gosm_options.solar_power_pos_threshold
        elif source.source_type == 'load':
            lower = gosm_options.load_power_neg_threshold
            upper = gosm_options.load_power_pos_threshold
        source.apply_bounds('forecasts', lower, upper)
        source.apply_bounds('actuals', lower, upper)
        source.to_csv(gosm_options.output_directory + os.sep + source.name)
    print("Done!")

if __name__ == '__main__':
    main()
