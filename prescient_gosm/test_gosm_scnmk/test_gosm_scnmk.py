import prescient_gosm.gosm_scnmk as scnmk
import os

if __name__ == '__main__':
    params = {"reference_model_file": os.path.join('..','..','models', 'knueven', 'ReferenceModel.py'),
              "start_date": '2013-01-01',"end_date": '2013-01-03', "sources_file": 'bpa_sourcelist.txt',
              "tree_template_file": os.path.join('..', '..', 'examples', 'gosm_test', 'TreeTemplate.dat'),
              "scenario_template_file": os.path.join('..', '..','examples', 'gosm_test', 'simple_nostorage_skeleton.dat')}

    scmaker = scnmk.Unitcommit_scenario_maker()

    scmaker.create_experiment("Experiment1", "augmentation", seed_offset=56)

    scmaker.set_parameters("Experiment1", params)

    scmaker.run_experiment("Experiment1", 2, "in_sample")
    scmaker.run_experiment("Experiment1", 4, "in_sample")

    scmaker.solve_experiment('Experiment1', 'in_sample')

    scmaker.create_experiment("Experiment2", "resampling", seed_offset=4)

    scmaker.set_parameters("Experiment2", params)

    scmaker.run_experiment("Experiment2", 2, "out_of_sample")
    scmaker.run_experiment("Experiment2", 4, "out_of_sample")

    scmaker.solve_experiment('Experiment2', 'out_of_sample')