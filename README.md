# Convolutional Neural Networks For Detection Of Pneumonia in Chest X-Ray Images


## Overview
Pneumonia is a lung condition characterized by inflammation primarily targeting the tiny air sacs called alveoli, present in one or both lungs. It can be triggered by viral or bacterial infections, and determining the specific pathogen responsible for causing Pneumonia can be exceedingly difficult.

The process of diagnosing Pneumonia typically begins with a thorough medical history and assessment of self-reported symptoms, followed by a physical examination that typically involves listening to the chest (chest auscultation). If the medical professionals suspect Pneumonia, they may recommend a chest radiograph. However, in adults who exhibit normal vital signs and a healthy lung examination, the likelihood of a Pneumonia diagnosis is low.

## Business Problem
Pneumonia continues to be a prevalent condition that poses significant morbidity and mortality risks. Annually, it affects around 450 million individuals and leads to approximately 4 million deaths. Early diagnosis plays a crucial role in improving patient outcomes; however, the conventional method of using radiographs often causes delays in diagnosis and treatment. Therefore, the development of a rapid and dependable computer-aided diagnosis system utilizing chest X-rays could be a crucial step toward enhancing outcomes for pneumonia patients.

In this project, I have created and evaluated various Convolutional Neural Networks (CNNs) capable of swiftly distinguishing between normal and pneumonia cases in frontal chest radiographs. The integration of these models could assist doctors and radiologists in identifying potential abnormal pulmonary patterns promptly, thereby expediting the diagnosis process.

## The Dataset
The data set was taken from [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code?datasetId=17810&sortBy=voteCount)
The dataset is divided into 2 directories:
train
test
Within each directory, there're 2 sub-directories:
NORMAL
PNEUMONIA

## Images:
Pre Model
![pre_model_imgs](https://github.com/jj862/Convolutional-Neural-Networks-For-Detection-Of-Pneumonia-in-Chest-X-Ray-Images/assets/69119958/24917c40-7ea2-4ccc-a205-b37d847393f4)

With Model_6
![lime_final_model](https://github.com/jj862/Convolutional-Neural-Networks-For-Detection-Of-Pneumonia-in-Chest-X-Ray-Images/assets/69119958/c15cfaa2-e7f3-467e-8ef4-f2ae20341c1d)




Glossary: 
* **Precision** - Indicates the proportion of positive identifications (model predicted class `1`) which were actually correct. A model which produces no false positives has a precision of 1.0.
* **Recall** - Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.
* **F1 score** - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.
* **Support** - The number of samples each metric was calculated on.
* **Accuracy** - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0, in other words, getting the prediction right 100% of the time.
* **Macro avg** - Short for macro average, the average precision, recall, and F1 score between classes. The Macro avg doesn't take class imbalance into effect. So if you do have class imbalances (more examples of one class than another), you should pay attention to this.
* **Weighted avg** - Short for the weighted average, the weighted average precision, recall, and F1 score between classes. Weighted means each metric is calculated with respect to how many samples there are in each class. This metric will favor the majority class (e.g. it will give a high value when one class outperforms another due to having more samples).

- False negatives — Model predicts negative, actually positive. In some cases, like email spam prediction, false negatives aren’t too much to worry about. But if a self-driving car's computer vision system predicts no pedestrian when there was one, this is not good.
- False positives — Model predicts positive, actually negative. Predicting someone has heart disease when they don’t, might seem okay. Better to be safe right? Not if it negatively affects the person’s lifestyle or sets them on a treatment plan they don’t need.
- True negatives — Model predicts negative, actually negative. This is good.
- True positives — Model predicts positive, actually positive. This is good.
- Precision — What proportion of positive predictions were actually correct? A model that produces no false positives has a precision of 1.0.
- Recall — What proportion of actual positives were predicted correctly? A model that produces no false negatives has a recall of 1.0.
- F1 score — A combination of precision and recall. The closer to 1.0, the better.
- Receiver operating characteristic (ROC) curve & Area under the curve (AUC) — The ROC curve is a plot comparing true positive and false positive rate. The AUC metric is the area under the ROC curve. A model whose predictions are 100% wrong has an AUC of 0.0, and one whose predictions are 100% right has an AUC of 1.0.

- 1. **R^2 (pronounced r-squared) or the coefficient of determination** - Compares your model's predictions to the mean of the targets. Values can range from negative infinity (a very poor model) to 1. For example, if all your model does is predict the mean of the targets, its R^2 value would be 0. And if your model perfectly predicts a range of numbers its R^2 value would be 1. 
2. **Mean absolute error (MAE)** - The average of the absolute differences between predictions and actual values. It gives you an idea of how wrong your predictions were.
3. **Mean squared error (MSE)** - The average squared differences between predictions and actual values. Squaring the errors removes negative errors. It also amplifies outliers (samples that have larger errors).

- For your regression models, you'll want to maximize R^2, whilst minimizing MAE and MSE.
