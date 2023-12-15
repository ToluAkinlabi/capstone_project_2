# Predicting Machine Failure
![selective focus photography of gray and brass colored metal parts](https://raw.githubusercontent.com/ToluAkinlabi/capstone_project_2/main/images/unsplash_header_image.jpg)
> Photo by [Jonathan Borba](https://unsplash.com/@jonathanborba) on [Unsplash](https://unsplash.com/photos/selective-focus-photography-of-gray-and-brass-colored-metal-parts-Hnws8oSFcgU)

## Project Overview
A single machine failure can grind an assembly line to a halt. If we can predict machine failures, then we can speed up the time to recovery.

### Navigating Repository:
* Exploratory work: Contains exploratory data analysis from all members
* Images: Visualization of graphs derived from the prediction
* Solutions.ipynb: Finalized notebook with the predictive analysis and algorithm
* README: Includes project overview, analysis, and recommendations
* [Presentation:](https://github.com/ToluAkinlabi/capstone_project_2/blob/main/presentation.pdf) Includes final PowerPoint presentation

## Business Understanding
**Problem:** Machines on an automotive assembly lines are interdependent upon each other in order to produce a complete automobile. If a machine in this line fails, it would delay the completion of the next steps in the assembly process, or it may produce defective parts increasing waste. Given sensor data, is it possible to detect a machine failure?<br>
**Stakeholders:** Auto manufacturers<br>
**Solution:** Using supervised machine learning, we attempt to predict manufacturing machine failures. This gives lead time for maintenance or replacement of the faulty machine.

## Data Understanding and Analysis

### Understanding sources of data
* [Machine Failure Prediction](https://www.kaggle.com/datasets/dineshmanikanta/machine-failure-predictions/data)

 1. A CSV file is obtained from Kaggle and within the file is a large repository of machine failure data including: Machine type, Air temperature, Process temperature, Roational speed, Torque, Tool wear (min), machine failure (binary), TWF(binary), HDF(binary), PWF(binary), OSF(binary), and RNF(binary).

### Data Analysis Process

 1. Collection: The data collection process is initiated by retrieving a csv file contained within kaggle. 
 
 2. Cleaning: The initial dataset was imbalanced, contained multiple failure types, and contained binary, categorical, and continuous data on vastly different scales. Resampling, scaling, binary transformation, and consolidation techniques were utilized to clean the data prior to processing.
 
 3. Processing: A host of different built-in python functions and specialized libraries were utilized with sklearn being the most utilized library for processing our data.
 
 4. Analysis: Employing visualization packages such as Matplotlib and Seaborn, we craft vivid representations of our models performance metrics.

## Model Evaluation
### Untuned model accuracy
![Untuned Model](https://raw.githubusercontent.com/ToluAkinlabi/capstone_project_2/main/images/untuned_model_accuracy_smooth_gradient.svg)

### Tuned model accuracy
![Tuned Model](https://raw.githubusercontent.com/ToluAkinlabi/capstone_project_2/main/images/tuned_model_accuracy.svg)


## Conclusion

**The recommendation for the top model in predicting machine failure will be Random Forest for these reasons:**

1. Decrease downtime
2. Enhanced Operational performance
3. Reduce maintenance cost
4. Increase product quality
5. Protect brand reputation

