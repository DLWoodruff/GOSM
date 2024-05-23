# prescient-wecc-data
Repository for run scripts and results associated with Prescient execution on the WECC-240 case.

To run these scripts/data in Prescient (current as of 03.20.18) (BR: brachun@sandia.gov)
1. Copy _everything_ from prescient-wecc-data/data to prescient/prescient/scripts (not the data/scenarios directory however)
2. Copy desired populator or simulator files from prescient-wecc-data/populator_scripts or prescient-wecc-data/simulator_scripts to prescient/prescient/scripts
3. From prescient/prescient/scripts run python runner.py <desired script file>
4. This should create an output folder in prescient/prescient/scripts. We are keeping those in prescient-wecc-data/prescient_outputs if you don't mind.

Reminder not to commit those scritps to the prescient repo.

