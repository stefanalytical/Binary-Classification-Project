# Binary-Classification-Project
This project is an end-to-end machine learning project that deals with classification algorithms utilizing [Weka 3.8.6](https://waikato.github.io/weka-site/index.html). The first portion of the project features the `diabetes.arff` dataset and the final portion features the `heart.csv` dataset. The `diabetes.arff` dataset contains Pima Indians onset of diabetes data. The `heart.csv` dataset is also a real life heart failure prediction dataset that contains attributes of patients who may or may not suffer from heart disease. All other files are different views of the original datasets.

## Project Goals

1.	Load the dataset.
2.	Analyze the dataset.
3.	Prepare views of the dataset.
4.	Evaluate algorithms.
5.	Finalize model and present results.

## Load the dataset

- Open the `diabetes.arff` dataset with the Weka Explorer.

<p float = "center">
  <img src="./PNGs/Figure1.png" alt="Getting started" width="500" height="400"/>
</p>

## Analyze the dataset

**Summary Statistics**

- This dataset has 768 instances, and since the models are evaluated using 10-fold cross-validation, then each fold has 76 instances.
- There are 9 attributes, with 8 being input and 1 being output.
- For the original `diabetes.arff` dataset, the input attributes are all numerical, but have different scales. The modified files feature the normalized and standardized data.
- There are no missing values.
- The class attribute is nominal and is a two-class or binary classification problem because it has two output values.
- The class attribute is also unbalanced because there is one *positive* outcome to 1.8 *negative* outcomes. There are nearly double the number of cases that are negative.

**Attribute Distributions**

<p float = "center">
  <img src="./PNGs/Figure2.png" alt="Getting started" width="500" height="400"/>
</p>

- There is a lot of overlap between the classes across the attribute values. This suggests that the classes are not easily separated.
- The class imbalance becomes evident when graphed (blue - tested_negative, red - tested_positive).
- Attributes plas, pres, skin, and mass have a Gaussian-like distribution.

**Attribute Interactions**

<p float = "center">
  <img src="./PNGs/Figure3.png" alt="Getting started" width="500" height="600"/>
</p>

- In general, there is poor separation between the classes on the scatter plots.
- It would be beneficial to transform the data and create multiple views by normalizing and standardizing.


**Prepare Views of the Dataset



