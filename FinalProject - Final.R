########################################################
### Final Project: Duke Men's Lacrosse
########################################################
rm(list=ls()) 
##setwd("C:/Users/User/Desktop/Fuqua/Fall I 2019/BusAnalytics/R Directory")
source("DataAnalyticsFunctions.R")
## read data into R

performance_data <- read.csv("LacrossePerformanceData.csv")

## Next use the function summary to inspect the data
summary(performance_data)


## Pull in some more packages
install.packages("gtools")
library(gtools)

## Make the ID, year, and season related columns factors since the numbers themselves are not relevant
performance_data$ID <- factor(performance_data$ID)
performance_data$Year <- factor(performance_data$Year)
performance_data$Season <- factor(performance_data$Season)
performance_data$ID.AND.SEASON <- factor(performance_data$ID.AND.SEASON)
summary(performance_data)


## drop columns that will cause level issues in prediction
data <- performance_data[,-c(1:4)]
## Create some calculated columns to have a metric that is not highly dependent on the player playing more time and having the opportunity to perform 
data$TurnoverEfficiency <- 100*data$Caused.Turnovers/max(data$Turnovers, 0.000001)
summary(data$TurnoverEfficiency)
data$StartPercentage <- 100*data$Games.Started/data$Games.Played
summary(data$StartPercentage)
data$PtsGame <- data$Goals...Assists/data$Games.Played
summary(data$PtsGame)

################################
## Create a holdout set from the data in performance data
## We decided to split the data randomly
## We will use the holdout to evaluate our model
## We chose to holdout 15% (32) because 10% (21) is too small given our dataset size
set.seed(1)
holdout.indices <- sample(nrow(data), 32)
data.holdout <- data[holdout.indices,]
data.train <- data[-holdout.indices,]
nrow(data.train)
nrow(data.holdout)

############################ 
## Now our data set is set up and ready to use
## First we want to check the correlation of some our data 
## Install and load necessary packages for a correlation plot
install.packages("corrplot")
library(corrplot)
## Plot our correlation on a correlation plot using circles
data_cor <- cor(data)
corrplot(data_cor[5:16,5:16], method="circle")
corrplot(data_cor[5:16,17:28], method="circle")
corrplot(data_cor[17:28,17:28], method="circle")

install.packages("GGally")
library(GGally)
mydata<-data[, c(13,14,19,28)]
ggpairs(mydata)
## Check some other plots
plot(data$November.BODY.WEIGHT, data$November.LIFT.TOTAL)
plot(data$Games.Started, data$November.POUND.4.POUND)
data_cor
## Data cleanup over
#####################
## Start creating a model to try and predict the new variables we created
## K Fold Cross Validation
## Set the random seed
set.seed(22)
### create a vector of fold memberships (random order)
n <- nrow(data)
nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
###
## Models to compare:
## model.f : with all variables
## Model.l : Lasso to select min choice of lambda
## model.pl: Post Lasso associated with Lasso and min choice of lambda
## Create a data frame to be used to test OOS performance
OOSPerformance <- data.frame(f=rep(NA,nfold), l=rep(NA,nfold), pl=rep(NA,nfold)) 

PerformanceMeasure <- function(prediction, actual, threshold) {
  mean( abs( prediction - actual )/ ifelse(abs(actual)>threshold, abs(actual),threshold) )  
  R2(y=actual, pred=prediction)
}

summary(data)
####################################
installpkg("glmnet")
library(glmnet)
#### Lets run Lasso
Mx<- model.matrix(PtsGame ~ ., data=data)[,-c(1,25)]
My<- data$PtsGame
lasso <- glmnet(Mx,My)
lassoCV <- cv.glmnet(Mx,My)
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(lassoCV, main="Fitting Graph for CV Lasso \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))
#### Post Lasso #####
installpkg("glmnet")
library(glmnet)
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
data.min <- data.frame(Mx[,features.min],My)

for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
## CV for the Post Lasso Estimates
pl <- glm(My~., data=data.min, subset=train)
predmin <- predict(pl, newdata=data[-train,], type="response")
OOSPerformance$pl[k] <- PerformanceMeasure(predmin, My, .02) 
## CV for the Lasso estimates  
lassomin  <- glmnet(Mx,My,lambda = lassoCV$lambda.min)
predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
OOSPerformance$l[k] <- PerformanceMeasure(predlassomin, My[-train], .02) 
OOSPerformance$l[k]  
### model.f : with full variables
model.f <- glm(PtsGame~., data=data, subset=train)
pred.f <- predict(model.f, newdata=data, subset=-train, type="response")
OOSPerformance$f[k] <- PerformanceMeasure(pred.f, My[-train], .02) 
###
print(paste("Iteration",k,"of",nfold,"completed"))
}

barplot(colMeans(OOSPerformance), las=2,xpd=FALSE , xlab="", ylab = bquote( "Average Out of Sample Performance (Relative)"))
## Lasso performs best in OOS performance compared to null and other models
## CV for the Lasso estimates  
lassomin  <- glmnet(Mx,My,lambda = lassoCV$lambda.min)
test.data <- data.holdout[,-25]

test.Mx<- model.matrix(~ ., data=test.data)[,-c(1,25)]
predlassomin <- predict(lassomin, newx=test.Mx, type="response")
predlassomin
##
##summary(test.data)
##summary(test.Mx)
##summary(data.holdout)
##length(test.Mx)
##length(test.data)
##
hist(predlassomin)
min(predlassomin)
### Compute the PCA
installpkg("glmnet")
library(glmnet)
x <- model.matrix(~., data=data)[,-1]
pca.x <- prcomp(x, scale=TRUE)
### Lets plot the variance that each component explains
plot(pca.x,main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)

### We can see that there are two main factor sets
### Lets plot so see what performance looks like in these components
points.pc <- predict(pca.x)
plot(points.pc[,1:2], pch=21,  main="")
text(points.pc[,1:2], labels=performance_data[,1], col="blue", cex=1)

##plot(points.pc[,3:4], pch=21,  main="")
##text(points.pc[,3:4], labels=performance_data[,1], col="blue", cex=1)
##################################################
## Interpreting the two factors
loadings <- pca.x$rotation[,1:2]
### For each factor lets display the top features that 
### are responsible for 3/4 of the squared norm of the loadings
v<-loadings[order(abs(loadings[,1]), decreasing=TRUE)[1:ncol(x)],1]
loadingfit <- lapply(1:ncol(x), function(k) ( t(v[1:k])%*%v[1:k] - 1/2 )^2)
v[1:which.min(loadingfit)]

v<-loadings[order(abs(loadings[,2]), decreasing=TRUE)[1:ncol(x)],2]
loadingfit <- lapply(1:ncol(x), function(k) ( t(v[1:k])%*%v[1:k] - 1/2 )^2)
v[1:which.min(loadingfit)]
###

