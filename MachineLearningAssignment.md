Building a Prediction Model for Quantified Self Movement
========================================================

### Executive Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

Data was collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

Using the accelerometer data, we build a prediction model that predicts the class of the barbell lift. The class represents a particular style of lift, with different styles having an effect on the effectiveness and safety of the exercsie. Random forests are used to create the final model, which has an out-of-sample accuracy of 99.5%.

### Loading and partitioning the data



First, we load the caret and randomForest libraries. 


```r
library(caret)
library(randomForest)
```

Then the training and test data sets are loaded. The training data for this project is available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data is available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
trainingSet <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!"))
testingSet <- read.csv ("pml-testing.csv")
set.seed(1)
```

We split the training data set into 2 parts:
* a training component used to train the model and
* a validation component used to estimate the out-of-sample error


```r
inTrain <- createDataPartition(trainingSet$classe, p = 0.6, list=FALSE)
training <- trainingSet[inTrain,]
validation <- trainingSet[-inTrain,]
```

We dedicate 60% of the data to training and 40% to validation.

### Preprocessing the data

We then remove metadata columns from the training set (timestamps, participant names etc.) that should not be used in the model. 


```r
# Remove metadata columns that I know shouldn't be part of the model
columnsToRemove <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")
columnsToKeep <- setdiff(names(training),columnsToRemove)
training <- training[,columnsToKeep]
```

We also remove columns that are sparsely populated at this point. We pick a threshold of 0.95 and decide to only consider columns where the fraction of null or NA values in the column is below the threshold. It would normally be sensible to remove columns with near-zero variance at this point, but this has already been done by removing the sparsely populated columns and therefore, has no effect in this case.


```r
# Remove columns that are mainly NA or have near-zero variance
populatedCols <- !apply(training, 2, function(x) (sum(is.na(x)) > (0.95 * dim(training)[1])))
training <- training[,populatedCols]
# training <- training[,!nearZeroVar(training)]
training$classe <- as.factor(training$classe)
```

The final training component has 53 columns (52 predictors) that we will use to train a model.

### Building a model

We then use the reduced training component to train a model. We choose to use random forests with the default number of trees.


```r
model <- randomForest(classe ~ ., data = training)
```

### Evaluating the in-sample error

Evaluation the in-sample error, we see that we have an accuracy of 100%. This is not surprising as the values that we are predicting were used to train the model. More important is the out-of-sample error.


```r
confusionMatrix(training$classe, predict(model, training))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

### Evaluating the out-of-sample error

Evaluating the out-of-sample error, we see that we have an accuracy of 99.5%. Therefore, for the 40% of data that was set aside and not used to train the model, we can predict the class of exercise very accurately. This is encouraging considering the fact that only one model was tested.


```r
confusionMatrix(validation$classe, predict(model, validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    4    1    0    1
##          B    6 1509    3    0    0
##          C    0    5 1360    3    0
##          D    0    0   15 1271    0
##          E    0    0    0    5 1437
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.993, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.994    0.986    0.994    0.999
## Specificity             0.999    0.999    0.999    0.998    0.999
## Pos Pred Value          0.997    0.994    0.994    0.988    0.997
## Neg Pred Value          0.999    0.999    0.997    0.999    1.000
## Prevalence              0.284    0.193    0.176    0.163    0.183
## Detection Rate          0.284    0.192    0.173    0.162    0.183
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.996    0.992    0.996    0.999
```
