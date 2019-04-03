
#Load training and test data
train = read.csv("train.csv")
test = read.csv("test.csv")


tweets <- data.frame(tweet = train$tweet)
tweets$Hatred = as.factor(train$label == 1)
table(tweets$Hatred)
tweets$tweet <- as.character(tweets$tweet)
####################################################################################
hateTweet = tweets$tweet[tweets$Hatred==TRUE]
hatecorpus = Corpus(VectorSource(hateTweet))
#Convert to lower casing
hatecorpus = tm_map(hatecorpus, tolower)

hatecorpus[[1]]$content

# Remove unidentified greek like characters
combined_doc <- function(x) iconv(x, 'utf-8', 'ascii', sub=' ')
hatecorpus = tm_map(corpus, content_transformer(combined_doc))
#look at stop words
stopwordlist = c(stopwords("en"), "user")
hatecorpus = tm_map(hatecorpus, removeWords, stopwordlist)
# remove punctation
hatecorpus = tm_map(hatecorpus, removePunctuation)
hatecorpus[[1]]$content
# Stem document
hatecorpus = tm_map(hatecorpus, stemDocument)
hatecorpus[[1]]$content
# Create document term matrix
hatefrequencies = DocumentTermMatrix(hatecorpus)
#Install Packages for text mining
#nstall.packages("wordcloud") # word-cloud generator 
#install.packages("RColorBrewer") # color palettes
library("wordcloud")
library("RColorBrewer")

# Build a term-document matrix
hatedtm <- TermDocumentMatrix(hatecorpus)
hatea <- removeSparseTerms(hatedtm, 0.995)
hatea
hatem <- as.matrix(hatea)

hatev <- sort(rowSums(hatem),decreasing=TRUE)
hated <- data.frame(word = names(hatev),freq=hatev)
hateh<-head(hated, 15)

# Word Cloud of all tweets
set.seed(1234)
wordcloud(words = hated$word, freq = hated$freq, min.freq = 1,
          max.words=5200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
# Bar plot
barplot(hated[1:25,]$freq, las = 2, names.arg = hated[1:25,]$word,
        col ="pink", main ="Most HATE frequent words",
        ylab = "Hate Word frequencies")

#######################################################################################

install.packages("tm")
library(tm)

install.packages("SnowballC")
library(SnowballC)

#Create Corpus
corpus = Corpus(VectorSource(tweets$tweet))

#View corpus
corpus
#View content of 1st
corpus[[1]]$content

#Convert to lower casing
corpus = tm_map(corpus, tolower)

corpus[[1]]$content

# Remove unidentified greek like characters
combined_doc <- function(x) iconv(x, 'utf-8', 'ascii', sub=' ')
corpus = tm_map(corpus, content_transformer(combined_doc))

#take out the hashtags


#look at stop words
stopwordlist = c(stopwords("en"), "user")
corpus = tm_map(corpus, removeWords, stopwordlist)

# remove punctation
corpus = tm_map(corpus, removePunctuation)
corpus[[1]]$content

# Stem document
corpus = tm_map(corpus, stemDocument)
corpus[[1]]$content



# Create document term matrix
frequencies = DocumentTermMatrix(corpus)


inspect(frequencies[1000:1015,505:555])



######################################################################################

install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
library("wordcloud")
library("RColorBrewer")

# Build a term-document matrix
dtm <- TermDocumentMatrix(corpus)
a <- removeSparseTerms(dtm, 0.995)
a
m <- as.matrix(a)

v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
h<-head(d, 15)

# Word Cloud of all tweets
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=5200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
# Bar plot
barplot(d[1:25,]$freq, las = 2, names.arg = d[1:25,]$word,
        col ="pink", main ="Most frequent words",
        ylab = "Word frequencies")
#####################################################################################

# Check for sparsity

findFreqTerms(frequencies, lowfreq=100)

# Remove sparse terms
sparse = removeSparseTerms(frequencies, 0.9976)
sparse
# Convert to a data frame

tweetsSparse = as.data.frame(as.matrix(sparse))

# Make all variable names R-friendly
colnames(tweetsSparse) = make.names(colnames(tweetsSparse))

# Add dependent variable

tweetsSparse$label = tweets$Hatred


# Split the data

library(caTools)

set.seed(12)

split = sample.split(tweetsSparse$label, SplitRatio = 0.7)

trainSparse = subset(tweetsSparse, split==TRUE)
testSparse = subset(tweetsSparse, split==FALSE)

####################################################################################
# Decision Tree
# Build a CART model

library(rpart)
install.packages("rpart.plot")
library(rpart.plot)

tweetCART = rpart(label ~ ., data=trainSparse, method="class")

prp(tweetCART)

# Evaluate the performance of the model
predictCART = predict(tweetCART, newdata=testSparse, type="class")


table(testSparse$label, predictCART)
library(caret)
confusionMatrix(table(testSparse$label, predictCART))


# Baseline accuracy 

table(testSparse$label)



#####################################################################################
# # Random forest model
# install.packages("randomForest")
# library(randomForest)
# 
# sparse = removeSparseTerms(frequencies, 0.992)
# sparse
# 
# set.seed(12)
# 
# library(caTools)
# split = sample.split(tweetsSparse$label, SplitRatio = 0.7)
# 
# trainSparse = subset(tweetsSparse, split==TRUE)
# testSparse = subset(tweetsSparse, split==FALSE)
# 
# # Training the trainingset Data with random forest 
# tweetRF = randomForest(label ~ ., data=trainSparse)
# 
# # Make predictions:
# predictRF = predict(tweetRF, newdata=testSparse)
# 
# #Confusion Marix
# confusionMatrix(testSparse$label, predictRF)
# # Accuracy:
# (293+21)/(293+7+34+21)
####################################################################################
# Logistic Regression
tweetLog = glm(label ~ ., data = trainSparse, family = "binomial")
summary(tweetLog)

predictions = predict(tweetLog, newdata=testSparse, type="response")

predictions = ifelse(predictions > 0.5, TRUE,FALSE)
library(caret)
confusionMatrix(table(testSparse$label, predictions))

