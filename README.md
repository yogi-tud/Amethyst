# Amethyst
Amethyst is a generalized on-the-fly de/re-compression framework for GPUs. Our developed framework allows to execute every computational function on every (un)compressed
input format and to output the result in an arbitrary (un)compressed format.

# Overview of repo

/src contains Amethyst source code

/visualisation contains additional visualisations

/data contains all experiment data

# Build and run experiments
To build and run all experiments:

./build.sh
./run.sh

The expected runtime should be between 1 and 5 hours.
Matrix experiments are excluded, as the needed runtime of 6 weeks is deemed too long for reproducibility.

Results are stored in /repro

# Paper


[Amethyst-A Generalized on-the-Fly De/Re-compression Framework to Accelerate Data-Intensive Integer Operations on GPUs,] (https://link.springer.com/chapter/10.1007/978-3-031-70626-4_8)

author={Fett, Johannes and Habich, Dirk and Lehner, Wolfgang},

booktitle={European Conference on Advances in Databases and Information Systems},
  
}
