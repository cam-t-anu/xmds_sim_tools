# xmds_sim_tools
Tools to help make xmds simulations more useful

## User Guide

### Prerequisites
* An xmds simulation with:
  * input parameters defined as arguments
  * \<arguments append_args_to_output_filename="yes"\> so that the analysis can parse out the input argunents
  * hdf5 output files with all the required data to evaluate your cost function
  * See 1DGEM-EIT.xmds for an example

### You will need to write
* A top level simulation running script that includes some or all of:
    * outer_sweep_settings - if you want to perform multiple optimisation runs over a range of values
    * inner_run_settings - these are paramters that help the framework run your simulation and includes the initial input values
    * analysis_settings - help the analyser parse the hdf5 output filenames, and evaluate the cost function
    * optimiser_settings - defines optimisation parameters, and other helpful parameters for optimisation 
    * see eit_gem_sims.py for an example format

### You will likely want to write
* Your own cost function calulator 
  * this can currently be written into analysis_sim_utils.py
  * otherwise you can use the efficeincy calculation that is already implemented
* A function to be evaluated at every outer sweep point
  * This is useful if you want to perform some calculation or operation after every optimisation
  * See outer_sweep_settings['outer_routine']

### You may want to write
* Your own optimiser
  * Currently the only included optimiser is optimise3.py
  * The software is designed such that other optimisers are easy to implement
  * Optimiser mu be a class, with support for API calls in sim_Optimiser.do_Round_Optimisation()


## General things TODO
* Write more optimisation algorithms, machine learning would be sweet
* Write better user and developer documentation (lol)
* Improve error handling
* Make the scripts into modules

