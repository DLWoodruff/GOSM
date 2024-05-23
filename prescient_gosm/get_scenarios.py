import sys
import prescient_gosm.gosm_options
import prescient_gosm.populator_options
import prescient_gosm.populator as populator


def createSrcFile(source, file_name):
    """
    This function was built as a helper function in order to create
    a sources file based on customizable sources which are
    formatted as dictionaries.
        Args:
            source (dict): A dictionary representing the source I want to
                insert into the file
            file_name (string): The name of the sourcelist file
    """
    if source is None:
        return


    #this appends to the file if it exists and creates it otherwise
    # Also make sure the file_name doesn;t already exist
    f = open(file_name, "a+")
    f.write("Source(" + source['name']+','+'\n')

    del source['name']

    for (i, j) in zip(range(len(source.keys())), source.keys()):
        if i == len(source.keys()) - 1:
            f.write("\t" + j + "=" + "\"" + source[j] + "\"" + ");"+"\n")
        else:
            f.write("\t" + j + "=" + "\"" + source[j] + "\"" + "," + "\n")

    f.close()

def parse_populator_ops(populator_ops):
    """
    Parse the dictionary containing what the user's options regarding the populator script and
    then set them as options that can be accessed when generating scenarios
        Args:
            populator_ops (dict): The populator script options which are set
                by the user

    """

    # Static dictionary containing default values for the options. These default values are from
    #   the populator_options.py file in presceient_gosm/presceient_gosm
    populator_dict = {'scenario_creator_options_file': None, 'start_date': None, 'end_date': None,
                      'sources_file': None, 'solar_scaling_factor': 1, 'wind_scaling_factor': 1,
                      'load_scaling_factor': 0.045, 'output_directory': None, 'allow_multiprocessing': 0,
                      'max_number_subprocesses': None, 'traceback': False, 'diurnal_pattern_file': None,
                      'number_dps': None,
                      'dps_paths_file': None, 'interpolate_data': False, 'average_sunrise_sunset': False}


    for operation, value in populator_ops.items():
        if operation not in populator_dict:
            raise ValueError(
                'This option is not defined for populator options. Refer to doc for options that are valid')
        else:
            populator_dict[operation] = value

    # Set the options for the populator_options file. This must be done in order to run the populator script
    for arg, value in populator_dict.items():
        setattr(sys.modules['prescient_gosm.populator_options'], arg, value)

    setattr(sys.modules['prescient_gosm.populator_options'], 'options_list', list(populator_dict.keys()))


def parse_sc_ops(scenario_ops):
    """
    Parse the dictionary containing what the user's options regarding the scenario_creator
    script and then set them as options that can be accessed when generating scenarios
        Args:
            scenario_ops (dict): The scenario_creator script options which are set
                by the user
    """

    # Static dictionary containing default values for the options. These default values are from
    #    the gosm_options.py file in presceient_gosm/presceient_gosm

    sc_dict = {'sources_file': None, 'output_directory': None, 'scenario_template_file': None,
               'tree_template_file': None, 'hyperrectangles_file': None, 'dps_file': None, 'reference_model_file': None,
               'load_scaling_factor': 1, 'scenario_day': None, 'planning_period_length': None,
               'historic_data_start': None, 'historic_data_end': None, 'use_temporal_copula': False,
               'sample_skeleton_points': False, 'number_scenarios': 10, 'separate_dps_paths_files': False,
               'solar_frac_nondispatch': 1, 'power_level_sunrise_sunset': 1, 'dps_sunrise': None, 'dps_sunset': None,
               'wind_frac_nondispatch': 1, 'cross_scenarios': True, 'use_separate_paths': False,
               'use_same_paths_across_correlated_sources': False, 'use_spatial_copula': False, 'spatial_copula': None,
               'partition_file': None, 'error_tolerance': 0, 'preprocessor_list': None,
               'solar_power_pos_threshold': None, 'solar_power_neg_threshold': None, 'wind_power_pos_threshold': None,
               'wind_power_neg_threshold': None, 'load_pos_threshold': None, 'load_neg_threshold': None, 'seg_N': 20,
               'epifit_error_norm': 'L2', 'seg_kappa': 100, 'probability_constraint_of_distributions': 1,
               'non_negativity_constraint_distributions': 0, 'nonlinear_solver': 'ipopt', 'L1Linf_solver': 'gurobi',
               'L2Norm_solver': 'gurobi', 'error_distribution_domain': '4,min,max', 'disable_plots': False,
               'temporal_copula': 'gaussian-copula', 'copula_prob_sum_tol': 0.01, 'plot_variable_gap': 10,
               'plot_pdf': 1, 'plot_cdf': 0, 'cdf_inverse_max_refinements': 10, 'cdf_tolerance': 0.0001,
               'cdf_inverse_tolerance': 0.0001, 'derivative_bounds': '0.3,0.7', 'monte_carlo_integration': False,
               'number_of_samples': 1000000, 'lower_medium_bound_pattern': -1000000000.0,
               'upper_medium_bound_pattern': 1000000000.0, 'granularity': 5.0, 'use_markov_chains': False,
               'error_bin_size': 100, 'use_error_quantiles': False, 'error_quantile_size': 0.1,
               'explicit_error_quantiles': None, 'consider_forecasts': False, 'forecast_bin_size': 100,
               'use_forecast_quantiles': False, 'forecast_quantile_size': 0.1, 'explicit_forecast_quantiles': None,
               'consider_derivatives': False, 'derivative_bin_size': 100, 'use_derivative_quantiles': False,
               'derivative_quantile_size': 0.1, 'explicit_derivative_quantiles': None, 'state_memory': 1,
               'use_equal_probability': False, 'seed': None, 'alpha': None, 'copula_random_walk': False}

    # Go through the user's dictionary and set the value for the defined dictionary above
    for operation, value in scenario_ops.items():
        if operation not in sc_dict:
            raise ValueError(
                'This option is not defined for scenario creator options. Refer to doc for options that are valid')
        else:
            sc_dict[operation] = value

    # Set the options for the gosm_options file. This must be done in order to run the populator script
    for arg, value in sc_dict.items():
        setattr(sys.modules['prescient_gosm.gosm_options'], arg, value)


def check_options():
    """
    There are certain options that must be set by the use and if the user did not
    set them, then we should return an error. This function is called after the
    user's options have been parsed and evaluated. Thus the populator_options
    and gosm_options should already be set. We will just get those values needed
    and if they are None or False they were not set by the user.

    """


    required_ops = []

    # These are required populator options
    required_ops.append(prescient_gosm.populator_options.output_directory)
    required_ops.append(prescient_gosm.populator_options.start_date)
    required_ops.append(prescient_gosm.populator_options.end_date)
    required_ops.append(prescient_gosm.populator_options.sources_file)

    # These are required scenario generator options
    required_ops.append(prescient_gosm.gosm_options.use_markov_chains)
    required_ops.append(prescient_gosm.gosm_options.copula_random_walk)
    required_ops.append(prescient_gosm.gosm_options.planning_period_length)
    required_ops.append(prescient_gosm.gosm_options.tree_template_file)
    required_ops.append(prescient_gosm.gosm_options.scenario_template_file)
    required_ops.append(prescient_gosm.gosm_options.reference_model_file)

    if (None in required_ops) or (False in required_ops):
        raise RuntimeError('You have not given one or more required options for scenario generation process. '
                           'Please refer to documentation for said options.')


def generate(populator_options, sc_options):
    """
    Reads the options data passed in by user and runs the populator.py script
    in order to generate scenarios defined by the options.

        Args:
            populator_options (dict): Stores options for the populator script.
                Contains the option parameter as the key and the value defined
                by the user. More info is found in the documentation.
            sc_options (dict): Stores options for the scenario_screator script
                which is called during the process of calling the populator
                script. Contains the option parameter as the key and the value
                defined by the user. More info is found in the documentation.

        Returns:
            path_to_output (str): After the scenarios are created return
                a string which is the path to the output data files
    """

    # Parse the options and initialize them
    parse_populator_ops(populator_options)
    parse_sc_ops(sc_options)

    # Check if user has given required options
    check_options()

    # Generate scenarios
    populator.populate()

    # Get path string to ouput
    path_to_output = prescient_gosm.populator_options.output_directory

    return path_to_output

