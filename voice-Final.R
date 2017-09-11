# Set up work directory
#------------------------------------------------------------------
setwd('D:/Dropbox/CS6140/Project/voice.csv')
# Read data
#------------------------------------------------------------------
gender <- read.csv("voice.csv")
set.seed(123)

# 1. Data preparation and exploration
# (a) Select the training set:
#-----Partition the dataset into a training and a validation subsets of equal size-----
trainNum <- sample(nrow(gender), floor(nrow(gender)* 0.5))
trainNum
train <- gender[trainNum,]
valid <- gender[-trainNum,]

# (b) Data exploration:
#one-variable summary statistics
# summary(train)

#two-variable summary statistics
# pairs(train)

# LDA ------------------------------------------------
library(MASS)
lda.fit = lda(label ~ ., data=train)
lda.fit
plot(lda.fit)

# ROC
#Tainning
library(ROCR)
scores <- predict(lda.fit, newdata= train)$posterior[,2]
pred <- prediction( scores, labels= train$label )
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=T, main="LDA")
# print out the area under the curve
unlist(attributes(performance(pred, "auc"))$y.values)

#Valid
scores <- predict(lda.fit, newdata= valid)$posterior[,2]
pred <- prediction( scores, labels= valid$label )
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=T, main="LDA")
# print out the area under the curve
unlist(attributes(performance(pred, "auc"))$y.values)

#SVM--------------------------------------------------------
library(caret)
library(ggplot2)
library(e1071)
control <- trainControl(method="cv", number=12)
metric <- "Accuracy"

# Linear Kernel
model.svm <- train(label~., data=train, method="svmLinear", metric=metric, trControl=control)
prediction.svm <- predict(model.svm, train)
confusionMatrix(prediction.svm, train$label)$overall[1]

model.svm <- train(label~., data=train, method="svmLinear", metric=metric, trControl=control)
prediction.svm <- predict(model.svm, valid)
confusionMatrix(prediction.svm, valid$label)$overall[1]

# Radial Kernel
model.svm <- train(label~., data=train, method="svmRadial", metric=metric, trControl=control)
prediction.svm <- predict(model.svm, train)
confusionMatrix(prediction.svm, train$label)$overall[1]

model.svm <- train(label~., data=train, method="svmRadial", metric=metric, trControl=control)
prediction.svm <- predict(model.svm, valid)
confusionMatrix(prediction.svm, valid$label)$overall[1]

# ROC
#Tainning
library(ROCR)
rocplot=function(pred, truth, ...){
  predob = prediction(pred, truth)
  pref = performance(predob, "tpr", "fpr")
  plot(pref,...)
}
svmfit.opt=svm(label~., data=train, kernel="radial", gamma=1, cost=1, decision.values=T)
fitted=attributes(predict(svmfit.opt, train, decision.values=TRUE))$decision.values
par(mfrow=c(1, 2))
rocplot(fitted, train$label, main="Training Data")
# print out the area under the curve
pred = prediction(fitted, train$label)
unlist(attributes(performance(pred, "auc"))$y.values)

#valid
fitted=attributes(predict(svmfit.opt, valid, decision.values=TRUE))$decision.values
rocplot(fitted, valid$label, main="Test Data")
pred = prediction(fitted, valid$label)
unlist(attributes(performance(pred, "auc"))$y.values)

