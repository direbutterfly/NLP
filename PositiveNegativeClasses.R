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
setwd("~/Desktop/CanadaLaw")
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

#ActsText=read_csv("ActsTextWOStopWords.csv")
#legalwords=c("act","na","order","regulation","subsection","amendment","section","repealed","sor","paragraph","regulations",
#             "definitions","NA","c s","a","the","shall","s","sections","subsections","i","ii","iii","column","item","-")
ActsText$Text=tolower(ActsText$Text)
badwords=c("repeal","repealed","c","s","repeals","r","—","act","regulation","order",
           "d","f","e","ii","iii","iv","v","vi","vii","ix","x","canada","b","section","subsection")
ActsText$Text = removeWords(ActsText$Text, badwords)
ActsText$Text = removeWords(ActsText$Text, stopwords("english"))
ActsText$Text =gsub("-|—","",ActsText$Text)
ActsText$ID=1:nrow(ActsText)
#ActsText$Text = removeWords(ActsText$Text, stopwords("english"))
#ActsText$Text = removeWords(ActsText$Text, legalwords)
#write.csv(ActsText,"ActsTextWOStopWords.csv")
#ActsText$Text=gsub(paste0(stopwords("english"),collapse = "|"),"",ActsText$Text,ignore.case = T)
#Positive
positive_data=ActsText[ActsText$Challenged==1,]
#positive$Text=ifelse(ActsText$Challenged==1,positive$Text,"")
corpus = VectorSource(positive_data$Text)
corpus_preproc = VCorpus(corpus)
corpus_preproc = tm_map(corpus_preproc,stripWhitespace)
corpus_preproc = tm_map(corpus_preproc,removePunctuation)
corpus_preproc = tm_map(corpus_preproc,removeNumbers)
#corpus_preproc = tm_map(corpus_preproc,content_transformer(tolower))
#corpus_preproc = tm_map(corpus_preproc,removeWords,stopwords("english"))
BigramTokenizer <- function(x) {RWeka::NGramTokenizer(x, RWeka::Weka_control(min=2, max=2))}
options(mc.cores=1)
dtm.pos.docs.2g <- DocumentTermMatrix(corpus_preproc, control=list(tokenize=BigramTokenizer))
dtm.pos.docs.2g.sp <- removeSparseTerms(dtm.pos.docs.2g, sparse = 0.98)
positive = as.data.frame(as.matrix(dtm.pos.docs.2g.sp))
positive_bigrams = colnames(positive.bigrams)

#Negative
negative_data=ActsText[ActsText$Challenged==0,]
#negative$Text=ifelse(ActsText$Challenged==0,negative$Text,"")
corpus = VectorSource(negative_data$Text)
corpus_preproc = VCorpus(corpus)
corpus_preproc = tm_map(corpus_preproc,stripWhitespace)
corpus_preproc = tm_map(corpus_preproc,removePunctuation)
corpus_preproc = tm_map(corpus_preproc,removeNumbers)
#corpus_preproc = tm_map(corpus_preproc,content_transformer(tolower))
#corpus_preproc = tm_map(corpus_preproc,removeWords,stopwords("english"))

dtm.neg.docs.2g <- DocumentTermMatrix(corpus_preproc, control=list(tokenize=BigramTokenizer))
dtm.neg.docs.2g.sp <- removeSparseTerms(dtm.neg.docs.2g, sparse = 0.997)
negative = as.data.frame(as.matrix(dtm.neg.docs.2g.sp))
negative_bigrams = colnames(negative)

##### Change the scores
neg.pos=negative[,colnames(negative) %in% positive_bigrams]
neg.pos=cbind.data.frame(neg.pos,ID=negative_data$ID)
neg.pos=merge(neg.pos,ActsText[,c("ID","ActName")],by="ID",all.y = T)
neg.pos[is.na(neg.pos)]=0

neg.neg=negative[,!(colnames(negative) %in% positive_bigrams)]*0.001
neg.neg=cbind.data.frame(neg.neg,ID=negative_data$ID)
neg.neg=merge(neg.neg,ActsText[,c("ID","ActName")],by="ID",all.y = T)
neg.neg[is.na(neg.neg)]=0

pos.neg=positive[,colnames(positive) %in% negative_bigrams]
pos.neg=cbind.data.frame(pos.neg,ID=positive_data$ID)
pos.neg=merge(pos.neg,ActsText[,c("ID","ActName")],by="ID",all.y = T)
pos.neg[is.na(pos.neg)]=0

pos.pos=positive[,!(colnames(positive) %in% negative_bigrams)]*999
pos.pos=cbind.data.frame(pos.pos,ID=positive_data$ID)
pos.pos=merge(pos.pos,ActsText[,c("ID","ActName")],by="ID",all.y = T)
pos.pos[is.na(pos.pos)]=0

neg.pos$ID=pos.neg$ID=neg.pos$ActName=pos.neg$ActName=NULL
#common=as.data.frame(ifelse(neg.pos!=0,pos.neg[,colnames(neg.pos)]/(neg.pos),pos.neg[,colnames(neg.pos)]))
common$ID=

#unique=rbind.data.frame(neg.neg,pos.pos)
features.ngrams=cbind.data.frame(neg.neg,pos.pos,common)
#write_csv(features.ngrams,"features.ngrams_sparse.csv")

#wordcloud
library(wordcloud)
source=negative[,!(colnames(negative) %in% positive_bigrams)]
challenged=0
source=positive[,!(colnames(positive) %in% negative_bigrams)]
challenged=1
a=cbind.data.frame(Text=ActsText$ActName,Challenged=ActsText$Challenged,source)
a=a[a$Challenged==challenged,]
b=a[a$Text=="Criminal Code",]
b$Challenged=NULL
b$Text=NULL
b=as.vector(t(b))
data=cbind.data.frame(labels=colnames(source),count=b)
data = data %>% arrange(desc(count))
data=data[1:50,]
wordcloud(data$labels, data$count, 
          min.freq =1, scale=c(5, .2), 
          random.order = FALSE, random.color = FALSE, 
          colors= c("lightsteelblue1","lightsteelblue2","lightsteelblue3","lightsteelblue"))

wordcloud(data$labels, data$count, min.freq =1, scale=c(5, .2), random.order = FALSE, random.color = FALSE, colors= c("indianred1","indianred2","indianred3","indianred"))
