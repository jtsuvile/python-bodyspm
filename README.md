## Python code for analysing body maps

This is a experimental version of python code that can be used instead of or in addition to (Matlab-based) BodySPM by Enrico Glerean (**add link**). 
It is designed to work with data coming from a version of https://version.aalto.fi/gitlab/eglerean/embody . 

### Functionality
This package takes the data from the online data collection system, and converts the topographies to .csv (with each cell representing one pixel). 
Additional subject info and info about the stimuli is saved as JSONs. These file formats are highly suitable for long term data storage and cross-platform compatibility.
There will also be an option to save the whole data set to a less sustainable data format for smooth analysis, as well as multiple basic data analysis options.

### TODO, general:
1. prep
    * edit read_bg to accept both list of filenames and single filename
    * check blur size for our data
2. analyses:
    * glm
3. visualise:
    * plot_analysis_results

### TODO, kipupotilaat:
1. Figure out how to save all controls into pickle (too large)
2. Find age and sex matched controls CRPS patients 
3. Compare CRPS patients to other pain patients
4. Controls: onko eroa täysin kivuttomilla ja akuuttia kipua kokevilla? Vastaako aiempia? (check with Lauri)
    
## Comments
Currently including group definitions at combine data phase, might make more sense to have that already when reading in subject?
