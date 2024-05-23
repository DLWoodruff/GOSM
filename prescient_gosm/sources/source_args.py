"""
source_args.py

This module will export a class SourceArgument which mirrors the structure of
an argument as defined in the builtin argparse module. This will also export
a list of arguments which will be recognized by the SourceParser.
"""

def boolean(string):
    """
    This returns True if the string is "True", False if "False",
    raises an Error otherwise. This is case insensitive.

    Args:
        string (str): A string specifying a boolean
    Returns:
        bool: Whether the string is True or False.
    """
    if string.lower() == 'true':
        return True
    elif string.lower() == 'false':
        return False
    else:
        raise ValueError("Expected string to be True or False")


def csv_file(string):
    """
    Returns the string if it ends in .csv, otherwise raises a ValueError.
    """
    if string.endswith('.csv'):
        return string
    else:
        raise ValueError("Expected string to end in .csv")


class SourceArgument:
    """
    Attributes:
        name (str): The name of the argument
        type (type): The type of the argument, a function which is called
            on the argument string to convert it to the appropriate type
        default: The default value of the argument
        required (bool): A flag indicating whether this argument is required
    """
    def __init__(self, name, type, default=None, required=False):
        """
        Args:
            name (str): The name of the argument
            type (type): The type of the argument, a function which is called
                on the argument string to convert it to the appropriate type
            default: The default value of the argument
            required (bool): A flag indicating whether an argument is required
        """
        self.name = name
        self.type = type
        self.default = default
        self.required = required

    def initialize(self, string):
        """
        This will initialize the argument and return the value of the argument
        after casting it as the appropriate the type.

        Args:
            string (str): The argument string
        Returns:
            The value of the argument
        """
        return self.type(string)

    def __repr__(self):
        return "SourceArgument({})".format(self.name)

    __str__ = __repr__


class SourceArgList:
    """

    """
    def __init__(self):
        self.args = {}

    def required_args(self):
        return [arg.name for arg in self.args.values() if arg.required]

    def add_argument(self, name, type, default=None, required=False):
        self.args[name] = SourceArgument(name, type, default, required)


source_args = SourceArgList()
source_args.add_argument('source_type', str, required=True)
source_args.add_argument('actuals_file', csv_file, required=True)
source_args.add_argument('forecasts_file', csv_file, required=True)
source_args.add_argument('segmentation_file', str)
source_args.add_argument('capacity_file', str)
source_args.add_argument('diurnal_pattern_file', csv_file)
source_args.add_argument('is_deterministic', boolean, default=False)
source_args.add_argument('frac_nondispatch', float, default=1)
source_args.add_argument('scaling_factor', float, default=1)
source_args.add_argument('forecasts_as_actuals', boolean, default=False)
source_args.add_argument('aggregate', boolean, default=False)
source_args.add_argument('disaggregation_file', str)
source_args.add_argument('time_step', str, default=False)
