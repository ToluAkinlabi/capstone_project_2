# Predicting Machine Failure
## Project Overview

### Navigating Repository:
* Exploratory work: Contains exploratory data analysis from all members
* Images: Visualization of graphs derived from the prediction
* Solutions.ipynb: Finalized notebook with the predictive analysis and algorithm
* README: Includes project overview, analysis, and recommendations
* [Presentation:](https://github.com/ToluAkinlabi/capstone_project_2/blob/main/presentation.pdf) Includes final PowerPoint presentation


## Project Overview


### Data Understanding and Analysis

#### Understanding sources of data
* [Machine Failure Prediction](https://www.kaggle.com/datasets/dineshmanikanta/machine-failure-predictions/data)

#### Data Analysis Process

 1. Collection: The data collection process is initiated by retrieving a csv file contained within kaggle. This csv file contains a large repository of machine failure data including: Machine type, Air temperature, Process temperature, Roational speed, Torque, Tool wear (min), machine failure (binary), TWF(binary), HDF(binary), PWF(binary), OSF(binary), and RNF(binary).
 
 2. Cleaning: The initial dataset was imbalanced, contained multiple failure types, and contained binary, categorical, and continuous data on vastly different scales. Resampling, scaling, binary transformation, and consolidation techniques were utilized to clean the data prior to processing.
 
 3. Processing: A host of different built-in python functions and specialized libraries were utilized with sklearn being the most utilized library for processing our data.
 
 4. Analysis: Employing visualization packages such as Matplotlib and Seaborn, we craft vivid representations of our models performance metrics.


#### Untuned model accuracy
![Untuned Model](https://raw.githubusercontent.com/ToluAkinlabi/capstone_project_2/main/images/untuned_model_accuracy_smooth_gradient.svg)

#### Tuned model accuracy
![Tuned Model](https://raw.githubusercontent.com/ToluAkinlabi/capstone_project_2/main/images/tuned_model_accuracy.svg)


## Conclusion

**The recommendation for the top model in predicting machine failure will be Random Forest for these reasons:**

1. Decrease downtime
2. Enhanced Operational performance
3. Reduce maintenance cost
4. Increase product quality
5. Protect brand reputation

#### Summary

