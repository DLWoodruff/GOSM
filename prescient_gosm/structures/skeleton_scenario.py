import datetime
import json
import os
from collections import OrderedDict, namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from . import skeleton_point_paths as paths
import prescient_gosm.pyspgen as pyspgen
import prescient_gosm.gosm_options as gosm_options
import prescient_gosm.basicclasses as basicclasses
from statdist.distribution_factory import distribution_factory
from statdist import UnivariateEpiSplineDistribution

load_key = 'Demand'
sources_key = 'MinNondispatchablePower MaxNondispatchablePower '


def disaggregate_dict(dict_, aggregate_source, disaggregated):
    """
    This method will update the dictionary of power values by replacing
    the values for the specified source by a collection of sources
    each with a proportion of the values.
    This will update the dictionry in-place.

    Args:
        dict_ (dict): The dictionary to disaggregate
        aggregate_source (str): The name of the source to be disaggregated
        disaggregated (dict[str,float]): A dictionary mapping names
            of the new sources to the proportion of the power of the
            original source
    """
    aggregated_power = dict_[aggregate_source]
    del dict_[aggregate_source]

    for name, proportion in disaggregated.items():
        source_power = [proportion*value for value in aggregated_power]
        dict_[name] = source_power


class SkeletonScenarioSet:
    """
    This class should manage all single skeleton scenarios and have
    methods for exporting data to scenario files as well.

    Attributes:
        scenarios (list[SkeletonScenario]): a list of scenarios
        actual_scenario (SkeletonScenario): the scenario from the actual data
        expected_scenario (SkeletonScenario): the scenario from the forecast
            data
        all_scenarios (list[SkeletonScenario]): The list of scenarios
            including the actual and expected scenario
    """

    def __init__(self, scenarios, actual=None, expected=None):
        """
        Initializes an object of the SkeletonScenarioSet class.

        Args:
            scenarios (List[SkeletonScenario]): The list of scenarios
            actual (SkeletonScenario): The actual scenario
            expected (SkeletonScenario): The expected scenario
        """

        self.scenarios = scenarios
        self.actual_scenario = actual
        self.expected_scenario = expected

        # We need the frequency of the underlying data for defining a suitable
        # index when writing out the raw scenarios.

        # Get the time step of the underlying data for all scenarios. It must
        # be the same for every scenario, so you can use the frequency of the
        # first scenario.
        self.frequency = scenarios[0].frequency

    @property
    def all_scenarios(self):
        """
        This property returns the list of probabilistic scenarios in addition
        to the actual scenario and the expected scenario.

        Returns:
            list[SkeletonScenario]: The list of all scenarios
        """
        return [self.actual_scenario, self.expected_scenario] + \
            sorted(self.scenarios)

    def write_raw_scenarios(self, directory, date):
        """
        This routine should write all of the raw scenario files to the
        directory specified. Raw refers to the fact that the file will only
        contain the n-vectors of the power generation and the probabilities.
        "n" refers to the number of time steps in the period, for which the
        scenario is created. This will create a file called 'scenarios.csv'
        in the directory specified. It is necessary to pass in the date since
        this object does not have any knowledge of the date of the scenario.

        Args:
            directory (str): The path to the directory to store the files
            date (datetime-like): The date of the scenarios
        """
        if not os.path.isdir(directory):
            os.mkdir(directory)

        scenario_frame = pd.DataFrame()
        index = ['Probabilty'] + list(
            pd.date_range(date, date + pd.Timedelta(gosm_options.planning_period_length),
                          freq=self.frequency))
        sources = list(self.scenarios[0].power_dict.keys())

        for source_name in sorted(sources):
            all_scenarios = self.all_scenarios
            for scenario in all_scenarios:
                if scenario.name == 'expected':
                    scen_name = 'forecasts'
                else:
                    scen_name = scenario.name

                scenario_name = source_name + ': ' + scen_name
                values = [scenario.probability] + \
                    scenario.power_dict[source_name]
                scenario_frame[scenario_name] = values
        scenario_frame.index = index
        scenario_frame.to_csv(directory + os.sep + 'scenarios.csv')

    def create_tree(self):
        """
        This creates an instance of the Scenario Tree class using
        self.scenarios.
        Returns:
            ScenarioTree: the scenario tree
        """
        root = InternalNode("root", probability=1)

        for scenario in self.scenarios:

            # We pass through the comments as well to the InternalNode
            # Questionable...
            internal_node = InternalNode(scenario.name, scenario.probability,
                                         scenario.data, root, scenario.comments)
            root.add_child(internal_node)

        tree = ScenarioTree()
        tree.set_root(root)

        return tree

    def normalize_probabilities(self):
        """
        This function will normalize the probabilities of the scenarios so
        that they add up to 1.
        """
        prob_sum = sum(scen.probability for scen in self.scenarios)
        for scen in self.scenarios:
            scen.probability /= prob_sum

    def normalize_names(self):
        """
        This function will change the names of the scenarios to be numbered
        in the form "Scenario_i".
        """
        for i, scenario in enumerate(self.scenarios):
            scenario.name = '{}'.format(i+1)

    def write_actual_and_expected(self, write_directory):
        """
        Writes json-files for the actual and forecast data.

        Args:
            write_directory: the directory to write in
        """

        actual_node = InternalNode(self.actual_scenario.name,
                                   self.actual_scenario.probability,
                                   self.actual_scenario.data)
        forecast_node = InternalNode(self.expected_scenario.name,
                                     self.expected_scenario.probability,
                                     self.expected_scenario.data)
        actual_node.write_json(write_directory)
        forecast_node.write_json(write_directory)

    def actual_and_expected_node(self):
        """
        Returns the corresponding Raw_Node_Data object for the actual and the
        expected scenario.

        Returns:
            (Raw_Node_Data, Raw_Node_Data): Actual, Expected Raw_Node_Data
        """
        return (self.actual_scenario.to_raw_node(),
                self.expected_scenario.to_raw_node())

    def plot_scenarios(self, directory, title, dps=None):
        """
        Basic plotting routine for the scenarios. This will create a
        plot for each source with all the power generation data for that
        given source.

        Args:
            directory (str): The name of the directory to save to
            title (str): The title of the plot
            dps (dict): the day part separators for each source if they are
                supposed to be in the plot
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # This is a little hack to get the source names, these are stored
        # as keys in the dictionary of a scenario
        sources = list(self.scenarios[0].power_dict.keys())

        # Create a plot for every source and add all scenarios.
        label = 'Scenarios'

        for source in sources:
            plt.figure(source)
            for scenario in self.scenarios:
                source_scenario = scenario.power_dict[source]
                plt.plot(source_scenario, 'k-', zorder=2, label=label,
                         marker='o', color='g')
                label = '_nolegend_'


            # Add forecast to the plot.
            if self.expected_scenario is not None:
                forecast_range = self.expected_scenario.power_dict[source]
                plt.plot(forecast_range, zorder=3, label='Forecast', color='r')

            if self.actual_scenario is not None:
                actual_range = self.actual_scenario.power_dict[source]
                plt.plot(actual_range, zorder=3, label='Actual', color='b')

            # Add dps to the plot.
            if dps is not None:
                label = 'Day Part Separators'

                for h in dps[source]:
                    plt.axvline(x=h, zorder=1, label=label,
                                color='grey', linestyle='--')
                    label = '_nolegend_'

            # Display a legend.
            lgd = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                             ncol=3, shadow=True)

            # Display a grid and the axes.
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.axvline(x=0, color='k')

            # Name the axes.
            plt.xlabel('Time')
            plt.ylabel('Power in Mw')

            # Create the suitable labels for the ticks at the x-Axes.
            locs, labels = plt.xticks()
            start = gosm_options.scenario_day
            freq = self.frequency
            new_locs = locs[1:]
            new_labels = []
            # We need to check if a number is included in the time step. If
            # yes, we need to compute the labels in a different way.
            if any(char.isdigit() for char in freq):
                for i in new_locs[:-1]:
                    timedelta = i * pd.Timedelta(freq)
                    time = start + timedelta
                    new_labels.append(time.strftime("%H:%M"))
            else:
                for i in new_locs[:-1]:
                    time = start + pd.Timedelta(str(i) + freq)
                    new_labels.append(time.strftime("%H:%M"))
            plt.xticks(new_locs, new_labels)

            # Create a title.
            plt.title(title + source, y=1.08)

            plt.savefig(directory + os.sep + source,
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close(source)


def merge_independent_scenarios(scenarios):
    """
    This creates a scenario which merges all the power dictionaries of the
    PowerScenario objects passed in. It will construct a name which is the
    concatenation of all scenario names, and a probability which is a product
    of all probabilities as we assume independence.

    Args:
        scenarios (List[PowerScenario]): The list of scenarios to merge
    Returns:
        PowerScenario: A scenario which is formed by merging all the other
            scenarios
    """

    name = ""
    power_dict = {}
    probability = 1
    comments = ''

    # We merge name, power dictionaries, probabilities, comments
    for scenario in scenarios:
        name += scenario.name + '_'
        power_dict.update(scenario.power_dict)
        probability *= scenario.probability
        if scenario.comments:
            comments += '\n' + scenario.comments


    # Here we drop the last underscore added
    name = name[:-1]

    return PowerScenario(name, power_dict, probability, comments)


# This will have a PowerScenario object and the corresponding paths
# used to create it. The paths attribute will point to a dictionary
# of the form {source_name -> OneDimPath}
ScenarioWithPaths = namedtuple('ScenarioWithPaths', ['scenario', 'paths'])


def merge_scenarios_with_paths(scenarios):
    """
    This will merge ScenarioWithPaths objects and return a ScenarioWithPaths
    objects which has the power generation vectors from all scenarios as well
    as the paths from all scenarios. We assume independence across the
    scenarios.

    Args:
        scenarios (list[ScenarioWithPaths]): A collection of ScenarioWithPaths
            objects to merge
    Returns:
        ScenarioWithPaths: The named tuple object with a merged PowerScenario
            and merged path dictionary
    """
    # We first merge the PowerScenario objects
    power_scenarios = [scen.scenario for scen in scenarios]
    scenario = merge_independent_scenarios(power_scenarios)

    # Then we merge their path dictionaries
    path_dict = {}
    for scen in scenarios:
        path_dict.update(scen.paths)
    return ScenarioWithPaths(scenario, path_dict)


class PowerScenario:
    """
    This class will only contain information about power generation and
    the associated probability and name. For each source of interest, this
    will store a n-vector of power-values produced. "n" refers to the number
    of time steps in the period, for which the scenario is created.

    Attributes:
        name (str): The name of the scenario
        power_dict (dict): A mapping from source names to lists of
            n floats of power generation over the day
        probability (float): A value between 0 and 1 representing the
            probability of the scenario
        comments (str): Additional details about how scenario was created
            among other things
    """
    def __init__(self, name, power_dict, prob, comments='', begin=None,
                 planning_period_length=None):
        """
        To initialize a PowerScenario object, one must pass a scenario name,
        a dictionary mapping source names to lists of n floats and an
        associated probability. "n" refers to the number of time steps in the
        period, for which the scenario is created.

        Args:
            name (str): The name of the scenario
            power_dict (dict[str,List[float]]): This is a dictionary mapping
                source names to a list of values
            prob (float): The associated probability of the scenario
            comments (str): Additional details about how scenario was created
                among other things
            begin (datetime like): The beginning of the first period of
                planing.
            planning_period_length (str): The planning period length. It is
                specified in the options file.
        """
        self.name = name
        self.power_dict = power_dict
        self.probability = prob
        self.comments = comments
        self.planning_period_length = planning_period_length
        self.begin = begin

        # The length of the vector representing the computed power values
        # for each scenario. It eqauls the number of time steps in the
        # period, for which the scenario is created.
        self.n = len(next(iter(power_dict.values())))

    def disaggregate_source(self, aggregate_source, disaggregated):
        """
        This method will update the dictionary of power values by replacing
        the values for the specified source by a collection of sources
        each with a proportion of the values.

        Args:
            aggregate_source (str): The name of the source to be disaggregated
            disaggregated (dict[str,float]): A dictionary mapping names
                of the new sources to the proportion of the power of the
                original source
        """
        disaggregate_dict(self.power_dict, aggregate_source, disaggregated)

    def aggregate_sources(self, source_names, aggregate_source):
        """
        This method will add up all the source power vectors for the sources
        provided and store that in a new source with the name aggregate_source.
        It will delete all the original source power vectors.

        Args:
            source_names (list[str]): Names of the sources to aggregate
            aggregate_sources (str): The name of the aggregate source
        """
        power_vector = [0]*self.n
        for name in source_names:
            for i, val in enumerate(self.power_dict[name]):
                power_vector[i] += val
            del self.power_dict[name]
        self.power_dict[aggregate_source] = power_vector

    def plot(self, axis=None):
        """
        Simple plotting routing which will plot all the power vectors
        for every source stored in this scenario onto the axis passed in
        (it will create one if none is passed in).
        Args:
            axis: The axis to plot to
        Returns:
            axis: The axis plotted to
        """
        if axis is None:
            fig, axis = plt.subplots()

        for name, vect in self.power_dict.items():
            xs = list(range(self.n))
            axis.plot(xs, vect, label=name)
        axis.set_xlabel('Hours of the Day')
        axis.set_ylabel('Power Values')
        axis.set_title('Scenario {}'.format(self.name))
        axis.legend()
        return axis

    def add_load_data(self, load_data, sources):
        """
        This will create a SkeletonScenario object using the data in the
        PowerScenario in conjunction with the load data passed in.

        Note this will not copy the values, so if they are changed by some
        other function, they will be changed in the newly created object

        Args:
            load_data (dict[str,List[float]]): A dictionary mapping names
                of load sources to n-vectors of load values. ( n refers to
                the number of time steps in the period, for which the
                scenario is created)
            sources (List[ExtendedSource]): A list of the sources used
                in the scenario
        Returns:
            SkeletonScenario: The scenario with power and load values
        """
        return SkeletonScenario(self.name, self.power_dict, self.probability,
                                load_data, sources, self.comments,
                                self.begin, self.planning_period_length)

    def truncate(self, source_name, lower=None, upper=None):
        """
        This method will truncate the source's power values at the passed
        in lower and upper bounds. If lower is not passed in, then no
        truncation will happen on the lower end. Similarly if upper is not
        passed in.

        Args:
            source_name (str): The name of the source to perform truncation
                               on.
            lower (float): The lower bound on power values.
            upper (float): The upper bound on power values.
        """
        for i, value in enumerate(self.power_dict[source_name]):
            if lower is not None and value < lower:
                self.power_dict[source_name][i] = lower
            elif upper is not None and value > upper:
                self.power_dict[source_name][i] = upper


    def __repr__(self):
        return "PowerScenario({})".format(self.name)

    def __str__(self):
        string = ""
        string += "PowerScenario({})\n".format(self.name)
        for source_name, power_vector in self.power_dict.items():
            string += "{}: {}\n".format(
                source_name, ", ".join(map(str, power_vector)))
        string += 'Probability: {}\n'.format(self.probability)
        return string

    def __lt__(self, other):
        return self.name < other.name


class SkeletonScenario(PowerScenario):
    """
    This class should contain all the data parameters and values that change
    from scenario to scenario (i.e, Min Dispatchable Power, Max Dispatchable
    Power). It will store these results in a dictionary called 'data'.
    """

    def __init__(self, name, power_dict, prob, load_data, sources,
                 comments='', begin= None, planning_period_length=None):
        """
        Initializes an object of the SkeletonScenario class.

        Args:
            power_dict (dict): a dictionary mapping source names to n-vectors
                of power generation values. ( n refers to the number of time
                steps in the period, for which the scenario is created)
            prob (float): the probability of the scenario
            load_data (List[float]): a n-vector of load-values ( n refers to
                the number of time steps in the period, for which the scenario
                is created)
            sources (List[ExtendedSource]): This is just used to get the source
                types
            comments (str): A string containing extra details about the
                scenario
            begin (datetime like): The beginning of the first period of
                planing.
            planning_period_length (str): The planning period length. It is
                specified in the options file.
        """
        PowerScenario.__init__(self, name, power_dict, prob, comments,
                               begin, planning_period_length)
        self.load_data = load_data

        self.types = {source.name: source.source_type for source in sources}
        self.dispatches = {source.name: source.frac_nondispatch
                           for source in sources}

        # A dictionary of data with strings as keys and the minimum and maximum
        # dispatch values as (str) values.
        self.data = {sources_key: OrderedDict(), load_key: OrderedDict()}

        # Get the time step from the underlying data. All sources must have
        # the same data frequency, so you can use the first source to get the
        # value.
        self.frequency = sources[0].time_step
        self.n = len(next(iter(power_dict.values())))
        self.sources = sources

        for i in range(self.n):
            for source in power_dict:
                # Translate the power generation values into strings of minimum
                # and maximum dispatch values.
                key = source + ' ' + str(i + 1)
                raw_value = power_dict[source][i]
                value = self.dispatch_value(self.dispatches[source], raw_value)
                self.data[sources_key][key] = value


        #print("N: ", self.n)

        for i in range(2 * self.n):
            for source in load_data:
                # Save the load forecast for both periods of planing.
                #print("Full load", load_data[source])
                #print("degub", i)
                #print("degub", load_data[source][i])
                forecast = load_data[source][i]
                key = source + ' ' + str(i + 1)
                self.data[load_key][key] = str(forecast) + '\n'

        # Copy the power generation values for the next time period of planing.
        #self.copy_power_generation()

        # Compute the power generation values for the next time period of
        # planing by using the deviation of forecast.
        self.deviation_of_forecast()

    def disaggregate_source(self, aggregate_source, disaggregated,
                            is_load=False):
        """
        This method will update the dictionary of power values by replacing
        the values for the specified source by a collection of sources
        each with a proportion of the values.

        Args:
            aggregate_source (str): The name of the source to be disaggregated
            disaggregated (dict[str,float]): A dictionary mapping names
                of the new sources to the proportion of the power of the
                original source
            is_load (bool): A flag to indicate whether the source to
                disaggregate is a load source
        """
        if is_load:
            power_dict = self.data[load_key]
            disaggregate_dict(self.load_data)
        else:
            PowerScenario.disaggregate_source(self, aggregate_source,
                                              disaggregated)
            power_dict = self.data[sources_key]

        for i in range(2*self.n):
            power_value = power_dict[aggregate_source+' '+str(i+1)]
            # These are stored as strings, so we convert to numbers and back
            power_value = map(float, power_value.split())
            del power_dict[aggregate_source+' '+str(i+1)]

            for name, proportion in disaggregated.items():
                power_value = [val * proportion for val in power_value]
                power_dict[name+' '+str(i+1)] = ' '.join(map(str, power_value)) + '\n'

    def write_raw_data(self, directory):
        """
        This function writes out the raw data for this scenario. The raw data
        in this sense refers to the n-vector of the power generation values
        produced in a scenario without any of the additonal pysp information.
        ( n refers to the number of time steps in the period, for which the
        scenario is created)

        The name of the file will be Scenario_<name>.dat where <name> is
        replaced by the name of the scenario.

        Args:
            directory (str): A path to the directory to store the scenario file
        """
        scen_file = directory + os.sep + 'Scenario_{}.dat'.format(self.name)

        with open(scen_file, 'w') as f:
            f.write('Probability: {}\n'.format(self.probability))
            for source in self.raw_data:
                f.write('Source: {}\n'.format(source))
                for dt, value in self.raw_data[source].items():
                    f.write('{},{}\n'.format(dt, value))

    def dispatch_value(self, dispatch, forecast):
        """
        Determines the minimum and the maximum dispatch value for the forecast.

        Args:
            dispatch (float): The fraction nondispatchable
            forecast (float): the forecast value

        Returns:
            string: the minimum and the maximum dispatch value, separated by a
                blank space
        """
        # In the case of solar power, the passed forecast will be None if the
        # respective hour lies outside the hours of sunshine.
        # In this case, set it to 0.
        forecast = 0 if forecast is None else forecast

        min_dispatch = dispatch * forecast
        value = "{} {}\n".format(min_dispatch, forecast)

        return value

    def deviation_of_forecast(self):
        """
        This function calculates the deviation of the last power generation
        value of the scenario from its forecast. This deviation is added
        to the forecasts of the 2. time period, so that we get power
        generation values for the 2. time period.
        """
        # First of all we need to get the last power generation value and the
        # last forecast of the first period for each source. These are used
        # to calculate the deviations for each source. We also need to get
        # all the forecasts for the 2. time period of planing.
        end = self.begin + pd.Timedelta(self.planning_period_length)
        end_of_second_period = end + pd.Timedelta(self.planning_period_length)
        date_range = pd.date_range(end, end_of_second_period,
                                   freq=self.frequency)
        last_values = {}
        last_forecasts = {}
        deviations = {}
        period2_forecasts = {}
        for source in self.power_dict:
            last_values[source] = self.power_dict[source][self.n - 1]
            for ex_source in self.sources:
                if ex_source.name == source:
                    last_forecasts[source] = ex_source.get('forecasts', end)
                    try:
                        period2_forecasts[source] = ex_source.get('forecasts',
                                                              date_range)
                    # If the 2. period of planing only contains one date, raise
                    # an exception if there is no data available for this date.
                    # Therefore a KeyError would appear and be handled by
                    # the error below.
                    except KeyError:
                        raise ValueError('There is no forecast available for '
                                         'some dates of the 2. period of '
                                         'planing. Please provide forecasts '
                                         'for the complete 2. period of '
                                         'planing.')
                    # If there are more dates in the 2. period of planing,
                    # the try statement above would not raise an exception, but
                    # it would return a dataframe with NaNs instead. So you
                    # have to check, if there are any NaNs in the list to
                    # raise an error just in case.
                    for x in list(period2_forecasts[source]):
                        if math.isnan(x):
                            raise ValueError(
                                'There is no forecast available for '
                                'some dates of the 2. period of '
                                'planing. Please provide forecasts '
                                'for the complete 2. period of '
                                'planing.')
                    break
            deviations[source] = last_values[source] - last_forecasts[source]

        # In the end the values for the 2. planing period are computed by
        # adding the deviation to the forecast values for each source. After
        # that they are added to self.data.
        for i in range(self.n):
            for source in self.power_dict:
                # Translate the power generation values into strings of minimum
                # and maximum dispatch values.
                key = source + ' ' + str(i + self.n + 1)
                raw_value = list(period2_forecasts[source])[i] + \
                            deviations[source]
                value = self.dispatch_value(self.dispatches[source], raw_value)
                self.data[sources_key][key] = value

    def copy_power_generation(self):
        """
        Copies the power generation data of the day for the next time period
        of planing, depending on the type of the respective source.
        """

        for i in range(self.n):
            for source, source_type in self.types.items():
                if source_type in ['solar', 'hydro']:
                    key = source + ' ' + str(i + 1)
                    value = self.data[sources_key][key]
                elif source_type in ['wind']:
                    key = source + ' ' + str(self.n)
                    value = self.data[sources_key][key]
                else:
                    raise RuntimeError("Power source '{}' has type '{}', "
                                       "the only types recognized are "
                                       "'solar', 'wind', and "
                                       "'hydro'.".format(source, source_type))
                key = source + ' ' + str(i + self.n + 1)
                self.data[sources_key][key] = value

    def to_raw_node(self):
        """
        Creates a daps-style Raw_Node_Data object from the scenario.
        Sets the parent to root currently.

        Returns:
            Raw_Node_Data: The equivalent Raw_Node_Data object
        """
        return pyspgen.CommentedRawNodeData(
            dictin=self.data, name=self.name, parentname='root',
            prob=self.probability, comments=self.comments)

    def __repr__(self):
        return "SkeletonScenario({})".format(self.name)

    def __str__(self):
        string = "SkeletonScenario({}):\n".format(self.name)
        for key, data in self.data.items():
            string += "{}:\n".format(key)
            for inner_key, inner_data in data.items():
                string += "{}: {}\n".format(inner_key, inner_data)
        return string


class ScenarioTree:
    """
    Basic Tree representation of a set of scenarios.
    The root points to an internal node which contains actual data for each
    stage.
    """
    def __init__(self):
        self.root = None

    def set_root(self, node):
        self.root = node

    def write_json_files(self, output_directory):
        """
        Writes json files for each of the scenarios in the tree
        """
        for child in self.root.children:
            child.write_json(output_directory)

    def create_raw_nodes(self):
        """
        This turns the scenarios stored in the true into daps-style
        Raw_Node_Data objects.

        Returns:
            (List[Raw_Node_Data]): A list of raw scenario nodes
        """

        return [child.to_raw_node() for child in self.root.children]


    def __str__(self):
        return "Tree:\n" + str(self.root)


class InternalNode:
    """
    Representation for an individual node in the Scenario tree.
    Each node has an associated name, probability, data,
    and pointers to parents and children.
    """

    def __init__(self, name, probability, data=None, parent=None, comments=''):
        """
        Initializes an object of the InternalNode class.

        Args:
            name (str): the name of the scenario
            probability (float): the probability of the scenario
            data: the data of the scenario
            parent: the parent node
        """
        self.name = name
        self.probability = probability
        self.parent = parent
        self.data = data
        self.children = []
        self.comments = comments

    def add_child(self, node):
        """
        Adds an internal node to the children list

        Args:
            node (InternalNode): An InternalNode object
        """
        self.children.append(node)

    def to_raw_node(self):
        """
        Converts the internal node into a daps-style Raw_Node_Data
        object.

        Returns:
            (Raw_Node_Data): raw node representing scenario
        """
        return pyspgen.CommentedRawNodeData(
            dictin=self.data, name=self.name, parentname=self.parent.name,
            prob=self.probability, comments=self.comments)

    def write_json(self, directory):
        """
        Writes json file for this node to the specified directory

        Args:
            directory: the directory to store the json file in
        """

        # if no parent specified, assume parent is root
        parent_name = 'root' if self.parent is None else self.parent.name

        filename = "NODE-{}-PARENT-{}-PROB-{}.json".format(
            self.name, parent_name, self.probability)
        with open(directory + os.sep + filename, 'w') as f:
            json.dump(self.data, f, sort_keys=True, indent=2)

    def __str__(self):
        string = "Internal Node {}:\nprobability: {}\ndata: {}\n".format(
            self.name, self.probability, self.data)
        string += 'Children:\n'
        for child in self.children:
            string += str(child)
        return string + '\n\n'
