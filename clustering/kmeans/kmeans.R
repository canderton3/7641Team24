library(ggplot2)
library(cluster)
library("factoextra")
# 2) elbow curve for each data set to find number of clusters to use

# 4) perform clustering 
# 5) do crossfold validation

# 1) read in file
data_exo <- read.csv('~/CS7641_Project/7641Team24/data/exoplanet.csv', header = TRUE, sep = ",")
data_exo_clean <- read.csv('~/CS7641_Project/7641Team24/data/exoplanet_cleanedrf.csv', header = TRUE, sep = ",")
data_exo_clean_score <- read.csv('~/CS7641_Project/7641Team24/data/exoplanet_cleanedrf_w_score.csv', header = TRUE, sep = ",")

# 2) Scale Data and Normalize
data_exo_clean_scaled <- scale(data_exo_clean, center=TRUE, scale = TRUE)
data_exo_clean_norm <-  (data_exo_clean_scaled -min(data_exo_clean_scaled))/(max(data_exo_clean_scaled)-min(data_exo_clean_scaled))

data_exo_clean_score_scaled <- scale(data_exo_clean_score, center=TRUE, scale = TRUE)
data_exo_clean_score_norm <-  (data_exo_clean_score_scaled-min(data_exo_clean_score_scaled))/(max(data_exo_clean_score_scaled)-min(data_exo_clean_score_scaled))

# 3) Find right number of clusters with the elbow curve. 
  # I found the cluster number to be between 8-10

#Elbow Method for finding the optimal number of clusters
# Compute and plot wss for k = 2 to k = 20.
# nstart option attempts multiple initial configurations and reports on the best one. 
  # For example, adding nstart=25 will generate 25 initial random centroids and choose the 
  # best one for the algorithm. Hope this helps!
  #https://datascience.stackexchange.com/questions/11485/k-means-in-r-usage-of-nstart-parameter

#data_exo_clean_scaled
k.max <- 20
data <- data_exo_clean_scaled
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     main = "Scaled Data Elbow Curve",
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#data_exo_clean_norm
k.max <- 20
data <- data_exo_clean_norm
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     main = "Scaled & Normoalized Data Elbow Curve",
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#data_exo_clean_score_scaled
k.max <- 20
data <- data_exo_clean_score_scaled
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     main = "Scaled Data Elbow Curve",
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#data_exo_clean_score_norm
k.max <- 20
data <- data_exo_clean_score_norm
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     main = "Scaled Data Elbow Curve",
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


# 4) cluster
k9 <- kmeans(data_exo_clean_score_scaled, centers = 8, nstart = 25)
exo_cluster <- fviz_cluster(k8, geom = "point",  data = data_exo_clean_score_scaled) + ggtitle("k = 9")
mydata <- data.frame(data_exo_clean_score_scaled, k9$cluster)

write.csv(mydata, '~/CS7641_Project/7641Team24/data/exoplanet_cleanedrf_w_score_w_cluster.csv')
