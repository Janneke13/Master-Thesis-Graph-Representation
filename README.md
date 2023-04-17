# Master Thesis- Structural Literal Representation
## By: Janneke van Baden

This GitHub repository contains files to run Graph Neural Network models with various structural literal representations.
Graph data must be given in either .nt or .nt.gz format. 

**A further explanation on how to run the code is provided below:**
<add explanation how to run it from the command line> <with the fixed names of the datasets>

The code itself is contained in the main directory. There are different files for all models, containing their classes.
These classes are used to create the models. Moreover, code to create the adjacency matrices with different layouts
are given, as well as code to run the models. 
A _requirements_ file is provided; these packages should be installed to run these experiments.

**The directories are structured as follows:**
- _data_ contains the datasets, which are all retrieved from https://gitlab.com/wxwilcke/mmkg. 
- _notebooks_ contains notebooks, for example for the EDA and the data analysis. PLots the results of the experiments.
- _plots_ contains the plots as generated by the notebooks, visualises different parts of the research.
- _results_ contains the files of the experiment results in .csv format.

