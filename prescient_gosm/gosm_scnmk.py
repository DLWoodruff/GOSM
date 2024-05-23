# ****  Gosm Scenario Maker ********
# DLW and Ravishdeep Singh - February 2019

import os
import shutil
import pandas as pd
import prescient_gosm.get_scenarios as gs
import pyomo.pysp.util.rapper as rapper
import pyomo.pysp.plugins.csvsolutionwriter as csvw


class UC_params:
    """
    Since there are multiple parameters used in order for a Unit Commit Scenario to be processed, this class will
    encapsulate all the data.

    Paramters:

    No matter if you are passing in your own scenarios or generating new one these two args are required!!!
    reference_model_file (string): Location of ReferenceModel.py.
    dates (Tuple(string)): Tuple containing start date and an end date which is the range for which the
                            scenarios will be constructed in.
    sources_file (string): Location of source file.
    tree_template_file (string): Location of tree template file.
    scenario_template_file (string): Location of scenario template file.
    planning_period_length (string): The length of time for the scenario period. The format is an integer
                                    followed by a capital letter, with "H" for hours and "T" for minutes.
                                    For example the default is "23H" representing a period for 24 hours.
    solvername (string): The name of the mathematical solver to use
    """

    def __init__(self, params={}):
        """
        Args:
            params (dict): the params dictionary is passed in and we add the arguments if they are found
                to each of the parameters. The attributes are as follows
        """

        self.reference_model_file = params.get("reference_model_file")
        self.start_date = params.get("start_date")
        self.end_date = params.get("end_date")
        self.sources_file = params.get("sources_file")
        self.tree_template_file = params.get("tree_template_file")
        self.scenario_template_file = params.get("scenario_template_file")

        self.planning_period_length = \
            (params.get("planning_period_length") if params.get("planning_period_length") is not None else '23H')
        self.solvername = (params.get("solvername") if params.get("solvername") is not None else 'gurobi')

    # These are setter functions for each of the parameters
    def set_reference_model(self, reference_model_file):
        self.reference_model_file = reference_model_file

    def set_dates(self, dates):
        (self.start_date, self.end_date) = dates

    def set_sources_file(self, sources_file):
        self.sources_file = sources_file

    def set_tree_template(self, tree_template_file):
        self.tree_template_file = tree_template_file

    def set_scenario_template_file(self, scenario_template_file):
        self.scenario_template_file = scenario_template_file

    def set_planning_period_length(self, planning_period_length):
        self.planning_period_length = planning_period_length

    def set_solver(self, solvername):
        self.solvername = solvername

    def __copy__(self):
        return UC_params(params={"reference_model_file": self.reference_model_file, "start_date": self.start_date,
                                 "end_date": self.end_date, "sources_file": self.sources_file,
                                 "tree_template_file": self.tree_template_file,
                                 "scenario_template_file": self.scenario_template_file,
                                 "planning_period_length": self.planning_period_length,
                                 "solvername": self.solvername})

    def check_valid_args(self):
        """
        If Reference Model file and dates are not specified then we can not run anything with these parameters
        Returns:
            (bool): True if the parameters can be used to run an experiment, False otherwise.

        """
        if (self.reference_model_file is None or self.start_date is None):
            print(self)
            raise RuntimeError("Essential Parameters are not set. To set, use method set_<parameter>(parameter)")

    # a str method for the class
    def __str__(self):
        return "reference_model_file: %s, dates: (%s, %s),sources_file: %s, tree_template_file: %s, " \
               "scenario_template_file: %s, planning_period_length: %s, solver: %s" \
               % (self.reference_model_file, self.start_date, self.end_date, self.sources_file, self.tree_template_file,
                  self.scenario_template_file, self.planning_period_length, self.solvername)


class Unitcommit_experiment:
    def __init__(self, name, sampling_method, seed_offset=0, params={}):
        """
        Initialize an Unitcommit_experiment instance which holds a current set of scenarios the user is testing with

        Args:
            name (string): the name given to this experiement
            seed_offset(int): number to start the seedlist
            sampling_method (string): The sampling method chosen for the iterations. The only two choices are
                resampling and augmentation
            params (dict): A dictionary  of user-defined parameters that will be used to initialize the UC_params class

        """

        self.name = name
        self.seedlist = [seed_offset]
        self.params = UC_params(params)
        self.sampling_method = sampling_method
        self.scenario_path = None

        # Number of scenarios ran so far in this experiment
        self.num_scenarios = 0

    def get_num_scenarios(self):
        """
        Getter function for num_scenarios
        """
        return self.num_scenarios

    def check_can_run(self):
        """
        Runs to see if the params passed in are viable to run an experiment with
        Returns:
            (bool): If the params are valid for a runnable instance
        """
        return self.params.check_valid_args

    def set_params(self, params):
        """
        Sets the parameters of the scenario by setting a UC_params class
        Args:
            params (dict): Dictionary of parameters which is passed into UC_params initializer

        """
        self.params = UC_params(params)

    def set_seed(self, seed):
        """
        Append a user picked seed into the seedlist.
        Args:
            seed (int): A seed that will be passed into the seedlist

        """
        self.seedlist.append(seed)

    def get_new_seed(self, additional_seeds):
        """
        Gets the seed for the next iteration in the procedure either by user-defined input or by sequential seedlist
            sequence for resampling. Augmentation needs only to keep track of the same seed during the procedure.
        Args:
            additional_seeds (List(int)): A list of seeds that the user might want to test with. Is not necessary.

        Returns: (int): seed used for the next iteration

        """
        if additional_seeds != []:
            self.seedlist += additional_seeds

        elif self.sampling_method == "resampling":
            self.seedlist.append(self.seedlist[-1] + 1)

        return self.seedlist[-1]

    def run_experiment(self, output_name, num_scenarios, additional_seeds=[]):
        """
        Run an iteration of the sequential sampling procedure on an experiment.
        Args:
            output_name (string): The location of the where the Scenarios folder will be located
            num_scenarios (int): The number of scenarios the user wants to get.
            additional_seeds (List(int)): A list of seeds that the user might want to test with. Is not necessary.

        """

        # Since iterations have non-decreasing sample sizes we must check if the number_scenarios follow so
        if num_scenarios < self.num_scenarios:
            raise RuntimeError("Error: Number of scenarios specified is less than what has already been incremented")

        self.num_scenarios += num_scenarios
        seed = self.get_new_seed(additional_seeds)


        Pop_opts = {'start_date': self.params.start_date, 'end_date': self.params.end_date,
                    'load_scaling_factor': 0.045,
                    'output_directory': os.path.join(output_name, 'Scenarios', self.name),
                    'sources_file': self.params.sources_file, 'allow_multiprocessing': 0, 'traceback': True}

        SC_opts = {'use_markov_chains': True, 'copula_random_walk': True,
                   'planning_period_length': self.params.planning_period_length,
                   'scenario_template_file': self.params.scenario_template_file,
                   'tree_template_file': self.params.tree_template_file,
                   'reference_model_file': self.params.reference_model_file, 'number_scenarios': self.num_scenarios,
                   'seed': seed}

        self.scenario_path = gs.generate(Pop_opts, SC_opts)

    def solve_scenarios(self, name, output_location):
        """
        Solves a set of scenarios.
        Args:
            name (string): Name of the experiment containing the scenarios to solve
            output_location (string): Location of where the solutions will be placed. Either in_sample or out_of_sample
        """
        solution_path = os.path.join(output_location, 'Solutions', name)

        # The scenario files have different folders representing the date range passed in during scenario generation
        date_range = pd.date_range(self.params.start_date, self.params.end_date, freq='D').strftime('%Y-%m-%d')

        # Reset the previous iteration's solutions if exists
        if os.path.isdir(solution_path):
            shutil.rmtree(solution_path)
        os.mkdir(solution_path)

        os.chdir(solution_path)

        # For each date folder solve the corresponding scenarios using the rapper API
        for date in date_range:
            tree_model_loc = os.path.join('..', '..', 'Scenarios', name, 'pyspdir_twostage', date,
                                          'ScenarioStructure.dat')
            reference_model_loc = os.path.join('..', '..', '..', self.params.reference_model_file)
            stsolver = rapper.StochSolver(reference_model_loc, tree_model=tree_model_loc)
            sopts = {'solution-writer': 'pyomo.pysp.plugins.csvsolutionwriter'}

            stsolver.solve_ef(self.params.solvername, sopts=sopts)
            csvw.write_csv_soln(stsolver.scenario_tree, "solution_" + date)

        os.chdir(os.path.join('..', '..', '..'))


class Unitcommit_scenario_maker:
    def __init__(self, in_sample_scenarios='In_Sample', out_of_sample_scenarios='Out_of_Sample'):
        """
        This class will be the interface to which the user will create experiments with
        Args:
            in_sample_scenarios (string): Name of where the in sample experiments will be placed
            out_of_sample_scenarios (string): Name of where the out of sample experiments will be placed
        """

        # Map of name to the actual UC_Experiment object. Thus we can not have two different experiments have
        #   the same name.
        self.UC_Experiments = {}

        if in_sample_scenarios == out_of_sample_scenarios:
            raise RuntimeWarning("Can not make in_sample and out_of_sample folders same name. "
                                 "Renaming to In_Sample and Out_of_Sample")
            self.in_sample_scenarios = 'In_Sample'
            self.out_of_sample_scenarios = 'Out_of_Sample'
        else:
            self.in_sample_scenarios = in_sample_scenarios
            self.out_of_sample_scenarios = out_of_sample_scenarios

        # Initialize directories for in_sample_scenarios and out_of_sample_scenarios
        self.initialize_directories()

    def initialize_directories(self):
        """
        This function initializes the directories for in_sample and out_of_sample directories if
            they are not found in the current directory originally

        """
        if not os.path.isdir(self.in_sample_scenarios):
            self.create_directories(self.in_sample_scenarios)

        if not os.path.isdir(self.out_of_sample_scenarios):
            self.create_directories(self.out_of_sample_scenarios)

    def create_directories(self, name):
        """
        Helper function that creates directories in current directory
        """
        os.makedirs(name)
        os.chdir(name)
        os.mkdir('Scenarios')
        os.mkdir('Solutions')
        os.chdir('..')

    def set_parameters(self, name, params):
        """
        Set the paramaters of the experiment for a certain scenario
        Args:
            name(string): The name of the scenario
            params (dict): Dictionary with parameters
        """
        if not self.check_experiment_exists(name):
            raise RuntimeError("Experiment with name %s has not been initialized yet" % (name));

        self.UC_Experiments.get(name).set_params(params)


    def copy_parameters(self, new_experiment, old_experiment):
        """
        Set the parameters for an experiment by copying the parameters of another experiment
        Args:
            new_experiment(string): The name of the experiment to initialize parameters
            old_experiment(string): Experiment with the parameters to copy from
        """
        if not self.check_experiment_exists(new_experiment):
            raise RuntimeError("Experiment with name %s has not been initialized yet" % (new_experiment));

        if not self.check_experiment_exists(old_experiment):
            raise RuntimeError("Experiment with name %s has not been initialized yet" % (old_experiment));

        self.UC_Experiments.get(new_experiment).set_params(self.UC_Experiments.get(old_experiment).params.copy())

    def create_experiment(self, experiment_name, sampling_procedure, seed_offset=0):
        """
        Creates an experiment that is stored in the scenario maker class
        Args:
            experiment_name (string): User-defined name of the experiment to be initialized
            sampling_procedure (string): Sampling procedure that the user wants for this experiment.
                Must be either resampling or augmentation.
            seed_offset(int): number to start the seedlist
        """
        if self.check_experiment_exists(experiment_name):
            raise RuntimeError("Experiment with that name already exists. Choose a new name");

        if (sampling_procedure != "resampling") and (sampling_procedure != "augmentation"):
            raise RuntimeError("Sampling procedure must be either 'resampling' or 'augmentation'")

        experiment = Unitcommit_experiment(experiment_name, sampling_procedure, seed_offset)
        self.UC_Experiments[experiment_name] = experiment

    def get_output_location(self, location):
        """
        Helper function to get the actual directory path given either 'in_sample' or 'out_of_sample'
        Args:
            location (string): Either 'in_sample' or 'out_of_sample' refering to the set of experiments in those locations

        Returns:
            (string) The name of the actual directory path. If the input was neither 'in_sample' or 'out_of_sample' we
                return None. The parent function can use this None to error check.

        """
        if location == "in_sample":
            return self.in_sample_scenarios
        elif location == "out_of_sample":
            return self.out_of_sample_scenarios
        else:
            return None

    def run_experiment(self, name, num_scenarios, location="in_sample", additional_seeds=[]):
        """
        Run an iteration of an experiment.
        Args:
            name (string): Name of the experiment to run.
            num_scenarios (int): The number of scenarios the user wants
            location (string): Directory location of where scenarios will be placed.
                Either 'in_sample' or 'out_of_sample'
            additional_seed (List(int)): A list of seeds that the user wants to test the experiment with

        """
        if not self.check_experiment_exists(name):
            raise RuntimeError("Experiment with name %s has not been initialized yet" % (name));

        output = self.get_output_location(location)

        if output is None:
            raise RuntimeError("Error: Must specifiy location of scenario. "
                               "The two options are in_sample and out_of_sample")

        self.UC_Experiments[name].run_experiment(output, num_scenarios, additional_seeds)

    def solve_experiment(self, name, location):
        """
        Solve a set of scenarios in an experiment.
        Args:
            name (string): Name of the experiment defined by the user
            location (string): Directory location of where scenarios of where scenarios were placed.
                Will be where the corresponding solutions will be placed. Either 'in_sample' or 'out_of_sample.
        """
        if not self.check_experiment_exists(name):
            raise RuntimeError("Error: Experiement with name %s does not exist." % (name))

        output = self.get_output_location(location)

        if output is None:
            raise RuntimeError("Error: Must specifiy location of scenario. "
                               "The two options are in_sample and out_of_sample")

        if not os.path.isdir(os.path.join(output, 'Scenarios', name)):
            raise RuntimeError("Did not create scenario of name: %s in %s location" % (name, location))

        self.UC_Experiments[name].solve_scenarios(name, output)

    def check_experiment_exists(self, name):
        """
        Checks if there is an experiment with the passed in name that already exists
        Args:
            name (string): Name of the experiment

        Returns:
            (bool): True if the experiment exists, False otherwise.
        """
        return self.UC_Experiments.get(name) is not None
