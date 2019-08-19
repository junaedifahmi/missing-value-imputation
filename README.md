# Missing Value Imputation Method

## Objective
This project is made because of my curiosity of how data mining works. The one problem that I see is missing value can affect the model we trying to create with some data mining algorithm. There are some sort of method that can solve this problem, like *mean-mode* method, where the missing value is replaces by their mean (for numerical attribute) or mode(for categorical attributes). The other solution for this problem is using imputation methods. Imputation methods, for short, is a technique that using model to predict what likely be the best value to replace that missing value with the respect of other data in the dataset.

## Proposed Method
In this works I split the missing attributes into numerical model (regression) and categorical model (classification). Although some algorithm can both be suitable for regression and classification, but in this work I want to find the best model for each attribute to works as they best for.

## Implementation
The implementation of the proposed method is using [WEKA](https://www.cs.waikato.ac.nz/ml/weka/) as data mining library. The data used for this experiments is consist of 7 dataset that can be found in [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The datasets are:
* Data 1
* Data 2
* Data 3

## Result
