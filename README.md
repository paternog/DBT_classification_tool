# DBT_classifier
DBT_classifier is a framework for the automatic classification of Digital Breast Tomosynthesis (DBT) images. It is composed of a set of a scripts and functions that enable the user to carry out the whole task, from the dataset preparation to the calculation of the relevant metrics and the activation maps useful for explainability purposes.

The slices of each DBT scan are classified in two ('Negative', 'Positive/lesion') or three classes ('Negative', 'Positive', 'Benign') independently form each other. Therefore, the core of the classification algorithm is a two-dimensional Convolutional Neural Network (CNN), choosen among different available architectures both standard and custom.

The main part of the framework is DBT_classifier.py script. It allows the user to choose and customize the CNN model and set a very high number of hyperparameters, among which those useful to define the training strategy (optimizer, epochs, batch_size, dropout, regularization, use of a DataGenerator, etc...). The training can be easily carried out on CPUs, a single GPU, or multiple GPUs, by just setting a few boolean variables. The script calculates automatically all the significant metrics for a classification task and makes the related plots. Finally, the GradCAM algorithm is applied to test images to obtain the activation maps.

DBT_classifier takes as input images in various format (in principle even DICOM). A sample of pre-processed images[1] of the public DUKE dataset[2] can be found at the following link: https://pandora.infn.it/public/56c90b.

The script dataset_creator_DUKE.py can be used to properly split the original data between training set and validation/test set, making sure to avoid data leackage. This can be repeated several times to obtain many training/test set pairs to perform a manual cross-validation of the model. Also, in oreder to try to mitigate overfitting, only a subset of slices for each DBT scan can be selected to be part of training and test sets. Finally, the script dataset_patient_folder_creator.py can be used to prepare the dataset for the so called "per patient analysis" (an approach similar to that adopted in [2]). Then, through the script load_and_test_model.py, which in general allows the user to assess the classification performance of a trained model, Free Responce Operating Curves (FROCs) reporting the true positive rate versus the number of false positive cases per scan can be obtained.

Finally, the script read_results.py can be used to aggregate the results obtained from models of the same CNN trained on different subsets (folds) of the original dataset. Whereas, the script read_results_analysis.py can be used to read the output data obtained from the previous one and it can be very useful to campare the performance of different CNN architectures trained on the same dataset.

Each script is organized in an elaboration ssection and an Input section, which is the only part the user has to modify according to what he wants to do.

[1] Esposito D, Paternò G, Ricciardi R, Sarno A, Russo P, Mettivier G (2024).
A pre-processing tool to increase performance of deep learning-based CAD in digital breast Tomosynthesis.
Health and Technology (2024) 14:81–91. https://doi.org/10.1007/s12553-023-00804-9

[2] Buda M, Saha A, Walsh R, Ghate S, Li N, Swiecicki A, Lo JY, Mazurowski MA (2021). A Data Set and
Deep Learning Algorithm for the Detection of Masses and Architectural Distortions in Digital Breast
Tomosynthesis Images. JAMA Netw Open 4:e2119100. https://doi.org/10.1001/jamanetworkopen.2021.19100

## Quick Start
We suggest to create a virtual environment with conda and install with pip the packages reported in the file "requirements.txt".

```
conda create --name name-you-want
pip install -r requirements.txt
```

## Version
version: 5.0, 
date: 18-03-2024

