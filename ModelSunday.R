library(data.table)
library(stringi)
library(dplyr)
library(readr)
library(slam)
library(tm)
library(ranger)
library(ROCR)
library(ggplot2)
library(pROC)
library(caret)

setwd("~/Desktop/CanadaLaw")

minmax = function(x){
  x=(x-min(x))/(max(x)-min(x))
}

Metrics = function(Actual,Prediction) {
  aucR <- roc(as.factor(Actual),Prediction)
  plot(aucR)
  cat("Cutoff Point: ",coords(aucR, "best")["threshold"])
  cm=data.frame(Actual=Actual,Prediction=ifelse(Prediction>coords(aucR, "best")["threshold"],1,0))
  print(table(cm))
  cm$Prediction=as.numeric(cm$Prediction) ; cm$Actual=as.numeric(as.character(cm$Actual))
  auc=pROC::auc(Actual,Prediction)
  recall=sum(cm$Actual*cm$Prediction)/sum(cm$Actual)
  precision=sum(cm$Actual*cm$Prediction)/sum(cm$Prediction)
  Fmeasure = 2 * precision * recall / (precision + recall)
  accuracy = sum(cm$Actual+cm$Prediction==0 | cm$Actual+cm$Prediction==2)/nrow(cm)
  return(list(cm=table(cm),metrics=cbind.data.frame(auc,recall,precision,Fmeasure,accuracy)))
}

#Act Names That got Challenged
SC=fread("SCacts.csv")
actsSC=SC[SC$response_var>0,"act_name"]

#Act Texts from XMLs
acts=read_csv("AllActs.csv")
acts$ActName=ifelse(is.na(acts$ActName),acts$DetailName,acts$ActName)
acts$SubSectionLabel=gsub("\\(|\\)| |[[:alpha:]]|*","",acts$SubSectionLabel)
acts$SectionLabel=gsub("\\(|\\)| |[[:alpha:]]|*","",acts$SectionLabel)
acts$SectionLabel=as.numeric(sub("-.*", "", acts$SectionLabel))
acts$SubSectionLabel=as.numeric(sub("-.*", "", acts$SubSectionLabel))
acts$SubSectionLabel=ifelse(is.na(acts$SubSectionLabel) | acts$SubSectionLabel=="NA",0,acts$SubSectionLabel)
acts$SectionLabel=ifelse(is.na(acts$SectionLabel) | acts$SectionLabel=="NA",0,acts$SectionLabel)

ActsText = acts %>% group_by(ActName) %>% summarise(Text=paste0(SectionText,SubSectionText,collapse = " "),
                                                    SectionCount=n_distinct(SectionName),
                                                    SubSectionCount=n_distinct(SubSectionName),
                                                    ExternalCitations=sum(n_distinct(c(SectionRefersExtAct,SubSectionRefersExtAct))))

ActsText$Text=gsub("NA","",ActsText$Text)
ActsText=as.data.frame(ActsText)

#Normalize the new features
ActsText$NofWords=minmax(stri_count_words(ActsText$Text))
ActsText$SectionCount=minmax(ActsText$SectionCount)
ActsText$SubSectionCount=minmax(ActsText$SubSectionCount)
ActsText$ExternalCitations=minmax(ActsText$ExternalCitations)

#Binary Response Variable
actsXML=unique(ActsText$ActName)
actnames = actsXML[actsXML %in% actsSC$act_name]
ActsText$Challenged=ifelse(ActsText$ActName %in% actnames,1,0)

#Combine ngrams with meta features and response variable
features.ngrams=fread("features.ngrams_sparse.csv")
features.ngrams=fread("features.bi.grams.sparse.csv")
data=cbind.data.frame(features.ngrams,Challenged=ActsText$Challenged,NoOfSections=ActsText$SectionCount,NoOfSubSec=ActsText$SubSectionCount,
                      WordCount=ActsText$NofWords,NoOfExtCitations=ActsText$ExternalCitations,ActName=ActsText$ActName)
colnames(data) <- make.names(colnames(data), unique=TRUE)

#Remove unnecessary features if present, remove big objects
data$ActName.1=data$ID=data$ID.1=data$ID.2=NULL
rm(features.ngrams)
rm(list = ls(pattern = "dtm|pos|neg")) 

#Create Train and "Final Unseen Test" Set
data$Challenged=factor(data$Challenged)
train.index <- createDataPartition(data$Challenged, p = .75, list = FALSE)
train <- data[ train.index,]
test  <- data[-train.index,]

#To circumvent memory error
options(expressions = 5e5)

##CV Ranger only on Train
#folds <- createFolds(train$Challenged, k=5,list = FALSE)
#train <- cbind.data.frame(train, folds)
folds <- createFolds(data$Challenged, k=5,list = FALSE)
train <- cbind.data.frame(data, folds)
predCV=c()
for (i in unique(folds)){
  trainCV=as.data.frame(train[train$folds!=i,])
  testCV=as.data.frame(train[train$folds==i,])
  trainCV$folds=testCV$folds=NULL
  weights=ifelse(trainCV$Challenged==1,10,1)
  modelCV=ranger(Challenged~.,trainCV[,!(colnames(trainCV) %in% c("ActName"))],classification=TRUE,probability = TRUE,num.trees = 1000,
                 mtry = 100,case.weights = weights)
  resultsCV=predict(modelCV,testCV[,!(colnames(testCV) %in% c("ActName"))],num.trees = modelCV$num.trees)
  predCV=rbind.data.frame(predCV,cbind.data.frame(Actual=as.factor(testCV$Challenged),Prediction=resultsCV$predictions[,2],
                                                  ActName=testCV$ActName))
}

#CV Metrics
resultsPositive=predCV[predCV$Actual==1,]
hist(resultsPositive$Prediction)
barplot(summary(cut(resultsPositive$Prediction, 5)), space=0)
predCV$RiskLevel=ifelse(predCV$Prediction>0.8,"High",
                        ifelse(predCV$Prediction>0.6,"Medium",
                               ifelse(predCV$Prediction>0.2,"Low","No")))

write.csv(predCV,"RiskLevelsAll.csv")
metricsCV=Metrics(predCV$Actual,predCV$Prediction)
metricsCV
fourfoldplot(margin.table(metricsCV$cm, c(1, 2)),color=c("red4","grey69"),margin = 2)

##Test set results
weights=ifelse(train$Challenged==1,10,1)
train$folds=NULL
model=ranger(Challenged~.,train,importance = "impurity",classification=TRUE,probability = TRUE,num.trees = 1000,
             mtry = 100,case.weights = weights) #,
results=predict(model,test,num.trees = model$num.trees)
#Test Metrics
metricsTest=Metrics(test$Challenged,results$predictions[,2])
metricsTest
fourfoldplot(margin.table(metricsTest$cm, c(1, 2)),color=c("red4","grey69"),margin = 2)

##Final model on the complete dataset
weights=ifelse(data$Challenged==1,1.5,1)
finalmodel=ranger(Challenged~.,data = data,importance = "impurity",classification=TRUE,probability = TRUE,num.trees = 1000,
                  mtry = 100 ,case.weights = weights)
finalpredictions=predict(finalmodel,data,num.trees = model$num.trees)
metricsTest=Metrics(data$Challenged,finalpredictions$predictions[,2])
metricsTest
results=cbind.data.frame(ActName=ActsText$ActName,Challenged=data$Challenged,Likelihood=finalpredictions$predictions[,2])
resultsPositive=results[results$Challenged==1,]
hist(resultsPositive$Likelihood)
varimp=cbind.data.frame(Feature=names(finalmodel$variable.importance),Imp=as.vector(finalmodel$variable.importance))
varimp = varimp %>% arrange(desc(varimp$Imp))

#Variable Importance Plot
ggplot(varimp[1:20,] , aes(x=reorder(Feature, Imp), y=Imp, fill=Imp),show.legend=FALSE) +
  geom_bar(stat='identity') +scale_fill_gradient(low="grey69",high="red4")+
  coord_flip() + xlab("Features") +ylab("Ranked by Importance") +
  theme(text = element_text(size=15),legend.position="none")
