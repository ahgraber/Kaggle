### Titanic survival classification

  # general setup
  library(tidyverse)
  
  
#--------------------------------------------------------------------------------------------------
### Read in data

  # Set working directory to the project folder
  setwd("~/Documents/Kaggle/Titanic")
  
  # Read in data
  train <- read.csv("train.csv")
  # test <- read.csv("test.csv")
  
  # # For Kaggle
  # train <- read.csv("../input/train.csv")
  # test <- read.csv("../input/test.csv")
  
#--------------------------------------------------------------------------------------------------
### Explore data
  
  str(train)
  summary(train)
  # NA's in age may present a problem... 177/891 are NA (20% of data!)
  
  # set up date
  rownames(train) <- train$PassengerId
  train <- train %>%
    select(-PassengerId, -Name, -Ticket, -Cabin) # Name, Ticket# shouldn't impact; Cabin has missing

  
  # clean factor (categorical) variables by creating a binary variable per category
  temp <- model.matrix(~ -1 + Embarked, data=train)
  train <- train %>%
    select(-Embarked) %>%  # remove original factor variables
    cbind(temp) %>%  # add new binary variables to the data set 
    select(-Embarked)  # remove Embarked column which has 1's for missing embarked locations           
  rm(temp)
  
  str(train)

#--------------------------------------------------    
### create two training files - one with Age NA's omitted; one with Age NA's filled
  
  # exclude NAs
  trainOmit <- na.exclude(train)
  
  # fill NAs
  # install.packages("VIM")
  library(VIM)
  trainFill <- kNN(train) %>%
    select(1:10)   # drops additional columns that kNN creates to identify cells to fill

#--------------------------------------------------------------------------------------------------
### Logistic regression
  names(train)
  
  
  logitOmit <- glm(Survived ~ ., family="binomial", data=trainOmit)
  summary(logitOmit)
  # Null deviance:  964.52
  # Resid deviance: 632.34
  # AIC: 652.34
  
  logitFill <- glm(Survived ~ ., family="binomial", data=trainFill)
  summary(logitFill)
  # Null deviance:  1186.66
  # Resid deviance: 767,66
  # AIC: 787.66
  
  confint(logitOmit)
  confint(logitFill)
  # Parch, Fare, EmbarkedC/Q/S don't seem to have much impact on either model
  
  # let's simplify both
  logitOmit2 <- glm(Survived ~ Pclass + Sex + Age + SibSp, family="binomial", data=trainOmit)
  summary(logitOmit2)
  # Null deviance:  964.52
  # Resid deviance: 636.72
  # AIC: 646.72
  
  logitFill2 <- glm(Survived ~ Pclass + Sex + Age + SibSp, family="binomial", data=trainFill)
  summary(logitFill2)
  # Null deviance:  1186.66
  # Resid deviance: 772.34,66
  # AIC: 782.34
  
  # and check against a 'cheat' backwards step...
  logitOmit3 <- step(logitOmit, direction="backward")
  summary(logitOmit3)
  # Null deviance:  964.52
  # Resid deviance: 633.22
  # AIC: 647.22
  # Keeps EmbarkedQ and EmbarkedS (I guess it doesn't need EmbarkedC b/c that's anyone else)
  # Drops EmbarkedC, Parch, and Fare
  
  logitFill3 <- step(logitFill, direction="backward")
  summary(logitFill3)
  # Null deviance:  1186.66
  # Resid deviance: 768.76
  # AIC: 780.76
  # Keeps EmbarkedS (but not one of Q or C???)
  # Drops EmbarkedQ, EmbarkedC, Parch, and Fare  
  
#--------------------------------------------------    
### Using Models
  # Let's move forward with the logitOmit model2 and model3 
  # logitOmit because we're not introducing error into our data with the kNN imputation
  # model2 and model3 because I'm curious how much EmbarkedS matters to overall accuracy
  
  # remove stuff we're not using
  rm(trainFill,logitFill,logitFill2,logitFill3)
  
  # what do our models predict?
  pred2 <- predict(logitOmit2, type="response")  # predicts for ALL in sample data
  pred3 <- predict(logitOmit3, type="response")
  summary(pred2)
  summary(pred3) 
  # remember, these give probability of classifying as a "1", where 1 = survived
  
  # let's turn the probability into a hard classification
  # borrowing/adapting function from class (Built by Matthew J. Schneider, Drexel LeBow):
  accuracy <- function(actuals, classifications){
    df <- data.frame(actuals, classifications);
    
    TP <- nrow(df[df$classifications==1 & df$actuals==1,]);  # true positive        
    FP <- nrow(df[df$classifications==1 & df$actuals==0,]);  # false positive
    FN <- nrow(df[df$classifications==0 & df$actuals==1,]);  # false negative
    TN <- nrow(df[df$classifications==0 & df$actuals==0,]);  # true negative
    
    precision <- round(TP/(TP+FP),3)  # positive hit rate
    accuracy <- round((TP+TN)/(TP+FN+FP+TN),3)  # overall accuracy
    tpr <- round(TP/(TP+FN),3)
    fpr <- round(FP/(FP+TN),3)
    scores <- c(precision,accuracy,tpr,fpr,TP,FP,FP,FN)
    names(scores) <- c("precision","accuracy","tpr","fpr","TP","TN","FP","FN")
    
    
    # scoresB=c(TP,FP,FP,FN)
    # names(scoresB)=c("TP","TN","FP","FN")
    # print(scoresB)
    
    return(list(scores));
  }
  
  # what probabilities from above model do we want to use as classification thresholds?
  # threshold <- c(.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)
  threshold <- c(.58, .59, .6, .61, .62)
  
  class2 <- list()
  for (i in 1:length(threshold)) {
    classify <- ifelse(pred2 > threshold[i], 1, 0)
    class2 <- append(class2, accuracy(trainOmit$Survived, classify))
  }

  class3 <- list()
  for (i in 1:length(threshold)) {
    classify <- ifelse(pred3 > threshold[i], 1, 0)
    class3 <- append(class3, accuracy(trainOmit$Survived, classify))
  }
  
  names(class2) <- threshold
  names(class3) <- threshold
  class2  # most accurate at .61 threshold
  class3  # most accurate at .6 threshold
  
  # model pred3 has a slightly better accuracy; we'll use it
  
  #--------------------------------------------------    
  ### predict on the testing set
  
  # first we have to set the testing data up the same way we did before
  test <- read.csv("test.csv")
  
  # for Kaggle
  # test <- read.csv("../input/test.csv")
  
  rownames(test) <- test$PassengerId
  test <- test %>%
    select(-Name, -Ticket, -Cabin) # Name, Ticket# shouldn't impact; Cabin has missing

  
  temp <- model.matrix(~ -1 + Embarked, data=test)
  test <- test %>%
    select(-Embarked) %>%  # remove original factor variables
    cbind(temp)  # add new binary variables to the data set 
  rm(temp)
  
  # even though we went with the NA.exclude data for train, we have to fill in data for test
  # or we won't get predictions for all rows
  # install.packages("VIM")
  library(VIM)
  test <- kNN(test) %>%
    select(1:10)   # drops additional columns that kNN creates to identify cells to fill

  # what if we use MICE to impute instead? (answer: we get the same score)
  # library(mice)
  # computeImpute <- mice(test, m = 5, 
  #                     method = vector("character", length = ncol(test)), 
  #                     predictorMatrix = (1 - diag(1, ncol(test))), 
  #                     visitSequence = "revmonotone", 
  #                     form = vector("character", length = ncol(test)), 
  #                     post = vector("character", length = ncol(test)), 
  #                     defaultMethod = "pmm", 
  #                     maxit = 5, diagnostics = TRUE, 
  #                     printFlag = F, 
  #                     data.init = NULL)
    
  # # transform the imputed data back into a data frame we can use
  # test <- complete(computeImpute)
  
  
  # use model to identify probability
  prediction <- predict(logitOmit3, newdata = test, type = "response")
  
  # use the same classificiation threshold (.6) that we used in the verification above
  survived <- ifelse(prediction > 0.60, 1, 0)
  
  out <- test %>%
    select(PassengerId) %>%
    cbind(survived)
    
  # save the output!
  write.csv(out, "TitanicLogicstic.csv", row.names=F)
  
#--------------------------------------------------------------------------------------------------
### SVM

  train <- read.csv("train.csv")
  rownames(train) <- train$PassengerId
  
  trainOmit <- na.exclude(train)
  
  # install.packages("e1071")
  library("e1071")
  
  svmOmit <- svm(Survived ~ ., data = trainOmit, type = "C", cost = 100000, epsilon = 0.1)
  # might be overfitting with this cost
  head(svmOmit$SV)
  predSVM <- predict(svmOmit, type="response")
  
  # # for testing new models against current best
  # svmOmit2 <- svm(Survived ~ Pclass + Sex + Age + SibSp, data = trainOmit, type = "C", 
  #                 cost = 20000, epsilon = 0.1)
  # predSVM2 <- predict(svmOmit2, type="response")
  
  accuracy(trainOmit$Survived, predSVM)
  # accuracy(trainOmit$Survived, predSVM2)
  
  #--------------------------------------------------    
  ### predict on the testing set
  
  # first we have to set the testing data up the same way we did before
  test <- read.csv("test.csv")
  
  # for Kaggle
  # test <- read.csv("../input/test.csv")
  
  rownames(test) <- test$PassengerId
  test <- test %>%
    select(-Name, -Ticket, -Cabin) # Name, Ticket# shouldn't impact; Cabin has missing

  
  temp <- model.matrix(~ -1 + Embarked, data=test)
  test <- test %>%
    select(-Embarked) %>%  # remove original factor variables
    cbind(temp)  # add new binary variables to the data set 
  rm(temp)
  
  # even though we went with the NA.exclude data for train, we have to fill in data for test
  # or we won't get predictions for all rows
  # install.packages("VIM")
  library(VIM)
  test <- kNN(test) %>%
    select(1:10)   # drops additional columns that kNN creates to identify cells to fill
  
  # use model to identify probability
  prediction <- predict(logitOmit3, newdata = test, type = "response")
  
  # use the same classificiation threshold (.6) that we used in the verification above
  survived <- ifelse(prediction > 0.60, 1, 0)
  
  out <- test %>%
    select(PassengerId) %>%
    cbind(survived)
    
  # save the output!
  write.csv(out, "TitanicSVM.csv", row.names=F)
  
  #### Notes:
  # SVM gives me the same rating on Kaggle that logistic regression does!

#--------------------------------------------------------------------------------------------------
### kNN

  train <- read.csv("train.csv")
  rownames(train) <- train$PassengerId
  
  trainOmit <- na.exclude(train)
  
  # consider binning Age, Fare
  hist(trainOmit$Age)
  hist(trainOmit$Fare)
  scaleAge <- scale(trainOmit$Age)
  scaleFare <- scale(trainOmit$Fare)
  
#--------------------------------------------------------------------------------------------------
### Decision tree / random forest

  train <- read.csv("train.csv")
  rownames(train) <- train$PassengerId
  
  trainOmit <- na.exclude(train)
  
  # install.packages("randomForest")
  library(randomForest)
  
  
  
  