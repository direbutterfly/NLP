library(data.table)
library(stringi)
library(dplyr)
library(readr)
library(slam)
library(tm)
library(ranger)
library(ROCR)
library(ggplot2)

minmax = function(x){
  x=(x-min(x))/(max(x)-min(x))
}

SC=fread("SCacts.csv")
actsSC=SC[SC$response_var>0,"act_name"]

acts=read_csv("AllActs.csv")
acts$ActName=ifelse(is.na(acts$ActName),acts$DetailName,acts$ActName)
acts$SubSectionLabel=gsub("\\(|\\)| |[[:alpha:]]|*","",acts$SubSectionLabel)
acts$SectionLabel=gsub("\\(|\\)| |[[:alpha:]]|*","",acts$SectionLabel)
acts$SectionLabel=as.numeric(sub("-.*", "", acts$SectionLabel))
acts$SubSectionLabel=as.numeric(sub("-.*", "", acts$SubSectionLabel))
acts$SubSectionLabel=ifelse(is.na(acts$SubSectionLabel) | acts$SubSectionLabel=="NA",0,acts$SubSectionLabel)
acts$SectionLabel=ifelse(is.na(acts$SectionLabel) | acts$SectionLabel=="NA",0,acts$SectionLabel)

ActsText = acts %>% group_by(ActName) %>% summarise(Text=paste0(SectionText,SubSectionText,
                                                                collapse = " "),
                                                    SectionCount=n_distinct(SectionName),
                                                    SubSectionCount=n_distinct(SubSectionName),
                                                    ExternalCitations=sum(n_distinct(c(SectionRefersExtAct,SubSectionRefersExtAct))),
                                                    MaxSectionNumber=max(SectionLabel,na.rm=T),
                                                    MaxSubSectionNumber=max(SubSectionLabel,na.rm=T))

ActsText$Text=gsub("NA","",ActsText$Text)
ActsText=as.data.frame(ActsText)

ActsText$NofWords=minmax(stri_count_words(ActsText$Text))
ActsText$SectionCount=minmax(ActsText$SectionCount)
ActsText$SubSectionCount=minmax(ActsText$SubSectionCount)
ActsText$ExternalCitations=minmax(ActsText$ExternalCitations)

actsXML=unique(ActsText$ActName)
actnames = actsXML[actsXML %in% actsSC$act_name]

ActsText$Challenged=ifelse(ActsText$ActName %in% actnames,1,0)

features.ngrams=fread("features.ngrams.csv")

data=cbind.data.frame(features.ngrams,Challenged=ActsText$Challenged,Sec=ActsText$SectionCount,SubSec=ActsText$SubSectionCount,
                      Count=ActsText$NofWords,Citations=ActsText$ExternalCitations)
colnames(data) <- make.names(colnames(data), unique=TRUE)

library(caret)
data$Challenged=as.factor(data$Challenged)
data$Challenged=ifelse(data$Challenged==0,"NotChallenged","Challenged")
train.index <- createDataPartition(data$Challenged, p = .7, list = FALSE)
train <- data[ train.index,]
test  <- data[-train.index,]

tc <- trainControl(method = "cv", 
                   number = 5 ,
                   classProbs = TRUE,
                   verboseIter = TRUE)

model <- train(Challenged~.,train ,trControl=tc, method="ranger", importance = 'impurity')
#model=ranger(Challenged~.,train,importance = "impurity")
results=predict(model,test)
confusionMatrix(results, test$Challenged)
varimp=cbind.data.frame(Feature=rownames(as.vector(varImp(model))$importance),Imp=as.vector(as.vector(varImp(model))$importance))


ggplot(varimp[1:20,] , aes(x=reorder(Feature, Overall), y=Overall, fill=Overall)) +
  geom_bar(stat='identity') +
  coord_flip() + xlab("Features") +ylab("Ranked by Importance")

perf1 <- performance(prediction(results,test$Challenged),"tpr","fpr")
plot(perf1)
#ss <- performance(prediction(results$predictions,test$Challenged), "sens", "spec")
ss <- performance(prediction(results,test$Challenged), "sens", "spec")
#performance(prediction(results$predictions,test$Challenged), "auc")@y.values
performance(prediction(results,test$Challenged), "auc")@y.values

#Let's evaluate confusion matrix on test set
#cm=data.frame(Actual=test$Challenged,Prediction=ifelse(results$predictions>ss@alpha.values[[1]][which.max(ss@x.values[[1]]+ss@y.values[[1]])],1,0))
cm=data.frame(Actual=test$Challenged,Prediction=ifelse(results>ss@alpha.values[[1]][which.max(ss@x.values[[1]]+ss@y.values[[1]])],1,0))

table(cm)
recall=sum(cm$Actual*cm$Prediction)/sum(cm$Actual)
precision=sum(cm$Actual*cm$Prediction)/sum(cm$Prediction)
Fmeasure = 2 * precision * recall / (precision + recall)
accuracy = sum(cm$Actual+cm$Prediction==0 | cm$Actual+cm$Prediction==2)/nrow(cm)
print(c(recall,precision,Fmeasure,accuracy))
#cm$ActName=ActsText$ActName

varimp=cbind.data.frame(Feature=names(model$variable.importance),Imp=as.vector(model$variable.importance))
varimp = varimp %>% arrange(desc(varimp$Imp))

ggplot(varimp[1:20,] , aes(x=reorder(Feature, Imp), y=Imp, fill=Imp)) +
  geom_bar(stat='identity') +
  coord_flip() + xlab("Features") +ylab("Ranked by Importance")

