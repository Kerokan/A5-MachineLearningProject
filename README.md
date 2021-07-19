# MachineLearningProjectA5

### Project made by Martin CAM and Sebastien CADUSSEAU
---

## Objectives

The objective of this project is to implement and test different algorithms of Machine Learning, and, in the end identify the best one. 

## Data description
### Dataset

The dataset contains 17898 candidates. Each candidate is described by 8 features and a single class. The class labels used are 0 (meaning that the candidate is NOT a pulsar) or 1 (meaning that the candidate is a pulsar).
The dataset is splitted in 2 sets for training and testing. The training set contains 8108 negative candidates (candidates that are NOT pulsars) and 821 positive candidates, for a total of 8929 candidates. The testing set contains 8151 negative candidates and 818 positive candidates, for a total of 8969 candidates.

[Link to the dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2)

### Features
As we said before, there are 8 features in this dataset :
- Mean of the integrated profile
- Standard deviation of the integrated profile
- Excess kurtosis of the integrated profile
- Skewness of the integrated profile
- Mean of the DM-SNR curve
- Standard deviation of the DM-SNR curve
- Excess kurtosis of the DM-SNR curve
- Skewness of the DM-SNR curve

All the features are continuous.
