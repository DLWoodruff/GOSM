"""
exceptions.py

This module should export any and all exceptions related to the process
of creating scenarios. The purpose of these exceptions is to make it clearer
to the user what problem is should for any single day, scenarios fail to be
created.

It is hopeful that these will be clearer than simply raising RuntimeError
at any problem.
"""


class InputError(Exception):
    def __init__(self, filename, data_type, message):
        self.data_type = data_type
        self.message = message


class DataError(InputError):
    def __init__(self, message, data_type):
        InputError.__init__(self, 'Data', message)


class HyperrectangleError(InputError):
    def __init__(self, message):
        InputError.__init__(self, 'Hyperrectangles', message)


class PathsError(InputError):
    def __init__(self, message):
        InputError.__init__(self, 'Paths', message)


class OptionsError(InputError):
    def __init__(self, message):
        InputError.__init__(self, 'Options', message)


class SegmentError(InputError):
    def __init__(self, message):
        InputError.__init__(self, 'Segmentation', message)


def print_input_error(error):
    print("INPUT ERROR:", file=sys.stderr)
    print("{} was raised.".format(error.__name__), file=sys.stderr)
    print("There is a problem with the {} file: {}".format(error.data_type,
                                                           error.filename),
          file=sys.stderr)
    print("'{}'".format(error.message), file=sys.stderr)

