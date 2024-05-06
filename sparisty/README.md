# Sparse Representations

This repository contains the code used to generate the sparsity boxplot (Figure N) in the Brain Sciences paper "Biologically-based computation: How neural details and dynamics are
suited for implementing a variety of algorithms". 

## Instructions

Clone or download the repository. 

The data has been split into 4 separate files so that they can be stored on github without exceeding the file-size limit. <br>
In order to create a single data file, open the `data_process.ipynb` notebook and run the cells at the bottom under the heading "Join Data Together". 

You should now have an npz file entitled "gridsearch_dataframe.npz" in your local folder. 
You can now run the `boxplot.ipynb` notebook to generate the boxplot. 
