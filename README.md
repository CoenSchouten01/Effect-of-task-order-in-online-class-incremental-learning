# Master Thesis: Investigating task order in Online Class Incremental Learning
This is the code repository for the Master's Thesis titled: Investigating task order in Online Class Incremental Learning.
The complete thesis is available here: *TODO*

This codebase, just like the thesis itself, contains two components: an analysis and curriculum designer evaluation.

## Analysis
To create the required logs for the analysis specify the strategies in "oracle_comparison.py" and run the following command, possibly accompanied by more arguments (options can be found under utils/argumentparser.py):

`python oracle_comparison.py --dataset {dataset} --num_classes {num_classes} --classes_per_task {classes_per_task} --num_runs {num_runs}`

The code for creating the figures and extracting the data for the tables for the analysis is gathered in the notebook "experiments_analysis.ipynb".
The full results can be found under "results/analysis".

## Curriculum Designer Evaluation
To create the required logs for the curriculum designer evaluation run the following command, possibly accompanied by more arguments (options can be found under utils/argumentparser.py):

`python cd_comparison.py --dataset {dataset} --strategy {strategy} --num_classes {num_classes} --classes_per_task {classes_per_task} --num_runs {num_runs}`

The code for creating the figures and extracting the data for the tables for the analysis is gathered in the notebook "experiments_cd_evaluation.ipynb".
The full results can be found under "results/cd_evaluation".