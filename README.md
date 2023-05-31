# Master Thesis- Structural Literal Representation
## By: Janneke van Baden

This GitHub repository contains files to run Graph Neural Network models with various structural literal representations.
Graph data must be given in either .nt or .nt.gz format. 

**A single experiment can be run using the following code:**
```commandline
python run_experiment.py
```

Which and how parameters can be set, can be seen using:
```commandline
python run_experiment.py -h
```

Alternatively, if the user wants to run 10 experiments with the same configuration in a row, with random seed from 
1 to 10, use:
```commandline
python run_final_experiments_loop.py
```

Or, if the user wants to do the same, but with inserted inverted relations, use:
```commandline
python run_final_experiments_loop_transposed.py
```

The paths currently implemented are for the four different datasets, as used in this thesis. 
They are: 'aifb', 'mutag', 'dmg777k', and 'synth'. Note that, before this is done, either the 
paths need to be changed to where the user stored the files, or the user needs to create a 'data' folder, as will be
specified below. The mapping techniques can be set with 'filtered', 'collapsed', 'all-to-one', and 'separate'.

***Creating a 'data' folder:***
The data folder has to be in the main directory, and contain four folders: 'aifb', 'mutag', 'dmg777k', and 'synth'. 
These folders themselves all contain a folder named 'gz_files', with the .gz files containing the triples, training,
validation, and test set. The data can be retrieved from: https://gitlab.com/wxwilcke/mmkg. 
For the full filepaths, the run_experiment.py file can be consulted.

***About the code:***

The code itself is contained in the main directory. There are different files for all models, containing their classes.
These classes are used to create the models. Moreover, code to create the adjacency matrices with different layouts
are given, as well as code to run the models. 
A _requirements_ file is provided; these packages should be installed to run these experiments.

**The directories are structured as follows:**
- _notebooks_ contains notebooks for the data analysis. Plots the results of the experiments.
- _results_ is created while running experiments, and contains the files of the experiment results in .csv format, 
as well as some plots used for preliminary analysis.
- _plots_ is created during the running of the notebooks, and contains the plots of the data analysis.