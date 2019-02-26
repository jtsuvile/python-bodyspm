## Python code for analysing body maps

This is a experimental version of python code that can be used instead of or in addition to (Matlab-based) BodySPM by Enrico Glerean (**add link**). 
It is designed to work with data coming from a version of https://version.aalto.fi/gitlab/eglerean/embody . 

### Functionality
This package takes the data from the online data collection system, and converts the topographies to .csv (with each cell representing one pixel). 
Additional subject info and info about the stimuli is saved as JSONs. These file formats are highly suitable for long term data storage and cross-platform compatibility.
There will also be an option to save the whole data set to a less sustainable data format for smooth analysis, as well as multiple basic data analysis options.

### TODO:
1. prep
    * combine_subwise_data 
2. analyses:
    * glm
    * ttest
    * compare_groups(master matrix, group definitions) ## pixel-wise test of proportions
3. visualise:
    * plot_analysis_results