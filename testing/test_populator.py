import os
import unittest
import pandas as pd
import time as t
import prescient_gosm.get_scenarios as gs


class GOSM_Test(unittest.TestCase):

    """
    The following four tests just tests if the software is able to run different energy sources
        The options can be user defined for tests but it is recommended to test using default
        options provided in tests before adding custom options to tests
    """
    def test_one_wind_source(self, Pop_opts = None, SC_opts = None):
        """
        Test for one wind source
        """

        if Pop_opts is None:
            Pop_opts = {'start_date': '2013-01-01', 'end_date': '2013-01-03', 'load_scaling_factor': 0.045,
                        'output_directory': 'examples/gosm_test/bpa_output',
                        'sources_file': 'testing/sourcelists_copies/bpa_sourcelist.txt', 'allow_multiprocessing': 0,
                        'traceback': True}

        if SC_opts is None:
            SC_opts = {'use_markov_chains': True, 'copula_random_walk': True, 'planning_period_length': '10H',
                       'scenario_template_file': 'examples/gosm_test/simple_nostorage_skeleton.dat',
                       'tree_template_file': 'examples/gosm_test/TreeTemplate.dat',
                       'reference_model_file': 'models/knueven/ReferenceModel.py', 'number_scenarios': 1,
                       'seed': 2}


        x = gs.generate(Pop_opts, SC_opts)
        self.assertIsNotNone(x)

        print("Output is located at ", x)

    def test_two_wind_sources(self, Pop_opts = None, SC_opts = None):
        """
        Testing multiple sources gosm with two wind source
        """

        if Pop_opts is None:
            Pop_opts = {'start_date': '2014-10-03-0:00', 'end_date': '2014-10-03-15:00', 'load_scaling_factor': 0.045,
                        'output_directory': 'examples/multiple_sources_gosm_test/CAISO/caiso_output',
                        'sources_file': 'testing/sourcelists_copies/caiso_sourcelist.txt', 'allow_multiprocessing': 0,
                        'traceback': True}

        if SC_opts is None:
            SC_opts = {'use_markov_chains': True, 'copula_random_walk': True, 'planning_period_length': '5H',
                       'scenario_template_file':
                           'examples/multiple_sources_gosm_test/CAISO/caiso_simple_nostorage_skeleton.dat',
                       'tree_template_file': 'examples/multiple_sources_gosm_test/TreeTemplate.dat',
                       'reference_model_file': 'models/knueven/ReferenceModel.py', 'number_scenarios': 5,
                       'seed': 2}


        x = gs.generate(Pop_opts, SC_opts)

        #Check if the output directory exists
        self.assertIsNotNone(x)

        # Test if the data files inside the directory are created since the directory can still be created
        #   but the scenario generation process can fail

        num_scenarios = SC_opts.get('number_scenarios') if SC_opts.get('number_scenarios') is not None else 10
        dates = pd.date_range(Pop_opts['start_date'], Pop_opts['end_date'])
        self.assertTrue(checkDirsExist(x, dates, num_scenarios))

        print("Output is located at ", x)

    def test_one_solar_source(self, Pop_opts = None, SC_opts = None):
        """
        Test for one solar source
        """

        if Pop_opts is None:
            Pop_opts = {'start_date': '2014-10-03-0:00', 'end_date': '2014-10-05-0:00', 'load_scaling_factor': 0.045,
                        'output_directory': 'examples/multiple_sources_gosm_test/CAISO/caiso_output',
                        'sources_file': 'testing/sourcelists_copies/casio_sourcelist2.txt',
                        'allow_multiprocessing': 0,
                        'traceback': True}

        if SC_opts is None:
            SC_opts = {'use_markov_chains': True, 'copula_random_walk': True, 'planning_period_length': '10H',
                       'scenario_template_file':
                           'examples/multiple_sources_gosm_test/CAISO/caiso_simple_nostorage_skeleton.dat',
                       'tree_template_file': 'examples/multiple_sources_gosm_test/TreeTemplate.dat',
                       'reference_model_file': 'models/knueven/ReferenceModel.py', 'number_scenarios': 1, 'seed': 10}

        x = gs.generate(Pop_opts, SC_opts)

        #Check if the output directory exists
        self.assertIsNotNone(x)

        # Test if the data files inside the directory are created since the directory can still be created
        #   but the scenario generation process can fail
        num_scenarios = SC_opts.get('number_scenarios') if SC_opts.get('number_scenarios') is not None else 10
        dates = pd.date_range(Pop_opts['start_date'], Pop_opts['end_date'])
        self.assertTrue(checkDirsExist(x, dates, num_scenarios))

        print("Output is located at ", x)


    def test_two_solar_sources(self, Pop_opts = None, SC_opts = None):
        """
            Testing multiple sources gosm with two solar sources.
            Using multiple solar sources is still a work in progress since the default options provide
                one of the few test cases that is able to provide scenarios for two solar sources.
            Thus it is recommended to not provide your own options until we are able to fix the
                issues with multiple solar sources
        """

        if Pop_opts is None:
            Pop_opts = {'start_date': '2014-10-03-0:00', 'end_date': '2014-10-03-0:00', 'load_scaling_factor': 0.045,
                        'output_directory': 'examples/multiple_sources_gosm_test/CAISO/caiso_output',
                        'sources_file': 'testing/sourcelists_copies/casio_sourcelist3.txt',
                        'allow_multiprocessing': 0,
                        'traceback': True}

        if SC_opts is None:
            SC_opts = {'use_markov_chains': True, 'copula_random_walk': True, 'planning_period_length': '23H',
                       'scenario_template_file':
                           'examples/multiple_sources_gosm_test/CAISO/caiso_simple_nostorage_skeleton.dat',
                       'tree_template_file': 'examples/multiple_sources_gosm_test/TreeTemplate.dat',
                       'reference_model_file': 'models/knueven/ReferenceModel.py', 'number_scenarios': 1, 'seed': 3}

        x = gs.generate(Pop_opts, SC_opts)

        self.assertIsNotNone(x)

        num_scenarios = SC_opts.get('number_scenarios') if SC_opts.get('number_scenarios') is not None else 10
        dates = pd.date_range(Pop_opts['start_date'], Pop_opts['end_date'])
        self.assertTrue(checkDirsExist(x, dates, num_scenarios))

        print("Output is located at ", x)


    def test_missing_required_opts_1(self):
        """
        Testing that the process returns an Error if the user doesn't
            provide one or more of the require options

        """

        Pop_opts = {'start_date': '2014-10-03-0:00', 'end_date': '2014-10-03-15:00', 'load_scaling_factor': 0.045,
                    'output_directory': 'examples/multiple_sources_gosm_test/CAISO/caiso_output',
                    'sources_file': 'testing/sourcelists_copies/caiso_sourcelist.txt', 'allow_multiprocessing': 0,
                    'traceback': True}

        SC_opts = {'use_markov_chains': True, 'copula_random_walk': True, 'planning_period_length': '10H',
                   'scenario_template_file':
                       'examples/multiple_sources_gosm_test/CAISO/caiso_simple_nostorage_skeleton.dat',
                   'tree_template_file': 'examples/multiple_sources_gosm_test/TreeTemplate.dat'}

        try:
            gs.generate(Pop_opts, SC_opts)
            self.assertEqual(True, False)
        except RuntimeError as e:
            self.assertEqual(True, True)


    def test_missing_required_opts_2(self):
        """
        Testing that the process returns an Error if the user doesn't
            provide one or more of the require options

        """

        Pop_opts = {'start_date': '2013-01-01', 'end_date': '2013-01-03', 'load_scaling_factor': 0.045,
                    'sources_file': 'testing/sourcelists_copies/bpa_sourcelist.txt', 'allow_multiprocessing': 0,
                    'traceback': True}

        SC_opts = {'use_markov_chains': True, 'copula_random_walk': True, 'planning_period_length': '10H',
                   'scenario_template_file':
                       'examples/multiple_sources_gosm_test/CAISO/caiso_simple_nostorage_skeleton.dat',
                   'tree_template_file': 'examples/multiple_sources_gosm_test/TreeTemplate.dat',
                   'reference_model_file': 'models/knueven/ReferenceModel.py', 'number_scenarios': 1}

        try:
            gs.generate(Pop_opts, SC_opts)
            self.assertEqual(True, False)
        except RuntimeError as e:
            self.assertEqual(True, True)



    def check_scenario_data(self):
        """
        Test that checks the scenario data. We have a copy of the data files that should
            be the output from this test in the directory 'baseline_scenario_data'. We
            will run the multiple wind sources test again and go through the new output
            files and if they are different from our baseline data files, which they shouldn't
            be, the test will fail.
        """

        # We need access to the output directory so we are defining the options outside the actual test
        Pop_opts = {'start_date': '2014-10-03-0:00', 'end_date': '2014-10-03-15:00', 'load_scaling_factor': 0.045,
                    'output_directory': 'examples/multiple_sources_gosm_test/CAISO/caiso_output',
                    'sources_file': 'testing/sourcelists_copies/caiso_sourcelist.txt', 'allow_multiprocessing': 0,
                    'traceback': True}

        self.test_two_wind_sources(Pop_opts=Pop_opts)


        data_files = ['Scenario_1.dat', 'Scenario_actuals.dat', 'Scenario_forecasts.dat',
                      'scenarios.csv', 'ScenarioStructure.dat']

        baseline_path = 'baseline_scenario_data/2014-10-03/'
        output_path = os.path.join(Pop_opts['output_directory'], 'pyspdir_twostage/2014-10-03/')

        for file in data_files:
            file1 = os.path.join(baseline_path,file)
            file2 = os.path.join(output_path, file)
            self.assertFalse(self.cmpfiles(file1, file2))

    def cmpfiles(self, file1, file2):
        """

        Args:
            file1 (string): Data file from the 'baseline_scenario_data'
            file2 (string): Data file from the output directory from the recent test run

        Returns:
            found_different (bool): Value dictating if we found that the files are different

        """
        f1 = open(file1)
        f2 = open(file2)
        found_different = False

        #Go through each line in each file and corresponding lines are
        #   different we must take a closer look at the line
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                self.check_equivalencies(line1.split(), line2.split(), found_different)
                if found_different:
                    break

        f1.close()
        f2.close()
        return found_different

    def check_equivalencies(self, list1, list2, found_different):
        """
        Args:
            list1 (list(string)): Line from data file from the 'baseline_scenario_data'
            list2 (list(string)): Line from data file from the output directory from the recent test run
            found_different (bool): Value dictating if we found that the files are different

        Returns:

        """
        for x, y in zip(list1, list2):
            if x != y:
                try:
                    #If the values differeing in the line are numbers we can allow them to
                    #   not be exactly equal and do not count as a discrepency between files

                    float_x = float(x)
                    float_y = float(y)
                    if not self.assertAlmostEqual(float_x, float_y, places=2):
                        found_different = True
                        break

                except ValueError:
                    #If the lines have a different discrepeny than the numbers then there
                    #   is definitly something wrong
                    found_different = True
                    break


def checkDirsExist(path, dates, num_scenarios):
    """
    Args:
        path (string): The path of the output directory specified
          by the user in the populator options
        dates (pd.DatetimeIndex): The date range derived from
            by the user in the populator options

    Returns:
        (bool): If the ouput directory has a directory
            corresponding to each date in the range
            we return True, otherwise False

    """

    # During the scenario generation process another directory is added on to the output directory path
    actual_path = os.path.join(path, 'pyspdir_twostage')

    for i in range(len(dates)):
        date = dates[i].date().strftime("%Y-%m-%d")
        date_path = os.path.join(actual_path, date)
        exists = os.path.exists(date_path)
        if not exists or not checkDirContents(date_path, num_scenarios):
            return False

    return True

def checkDirContents(date_path, num_scenarios):
    """
    Args:
        date_path (string): The date which corresponds to the directory
            containing the date files corresponding to the date's scenarios
        num_scenarios (int): The number of scenarios since there will be that many
            scenario data files in each date directory

    Returns:
        has_contents (bool): Each date directory has the corresponding
            scenario data files

    """

    scenario = "/Scenario"

    # These files are in every directory no matter the number of scenario data files
    has_actuals = os.path.isfile(date_path + scenario + '_actuals.dat')
    has_forecasts = os.path.isfile(date_path + scenario + '_forecasts.dat')
    has_structure = os.path.isfile(date_path + scenario + 'Structure.dat')

    has_contents = has_actuals and has_forecasts and has_structure

    # Go through and there are the same number of scenario data files as specified
    for i in range(num_scenarios):
        if not os.path.isfile(date_path + scenario + '_' + str(i + 1) + '.dat'):
            has_contents = False
            break

    return has_contents



if __name__ == '__main__':
    unittest.main()()
