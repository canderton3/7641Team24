# Introduction/Background
NASA built and launched the Kepler Space Observatory satellite in 2009. The primary objective of the telescope was to find other planets outside of our solar system, known as exoplanets; to do this, Kepler collects data on "objects of interest" that could be classified as exoplanets based on certain characteristics. Despite some mechanical failures, Kepler is still active and continues to gather data on possible planets outside of our solar system. By discovering new exoplanets in our galaxy, we can continue humanity's search for life on other planets and possible habitable zones.

# Problem definition
Using the data that Kepler has collected, NASA is trying to classify exoplanets as candidates for habitation, or as false positives. As there are a nearly infinite number of “objects of interest” in the galaxy let alone, the universe, the amount of time to classify these objects of interest as probable habitable planets versus non-habitable would be too costly and time-intensive for humans to do. Therefore, we are creating a machine learning model to predict/classify whether an object of interest detected by Kepler is a candidate to be classified as a planet, to reduce labor costs and human effort. In the process of creating this model, we can also discover key characteristics that Kepler measures that are indicative of an exoplanet, which could eventually narrow our search even further and make it less computationally expensive to identify exoplanets.

# Data Collection
## Exoplanet Dataset
The dataset was sourced from Kaggle, linked [here](https://www.kaggle.com/nasa/kepler-exoplanet-search-results), sourced directly from NASA. The dataset is a record of all of the "objects of interest" that Kepler has collected data on. The inital size was 3.52 MB, with 50 columns and around 10,000 data points. Each data point contains information about physical characteristics of already classified objects identified by Kepler; the data point also has a label (koi_pdisposition) designating it as either a CANDIDATE or a FALSE POSITVE. Our goal is to identify CANDIDATE planets.

Thankfully the dataset came with a thorough data dictionary, linked [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html). Our goal was to use the possible exoplanet's physical characteristics to identify the object of interest as either a candidate or false positive, so we we focused mostly on the physical characteristics columns. More information about feature selection is below.

## Data Cleaning and Preparation
This dataset was relatively easy to clean-there was no aggregation to be done or significant issues to handle. The main issues were NaN values and feature selection.

### Cleaning
Before dropping any columns, 36 columns contained at least some NaN values. This obviously was not going to be helpful for our purposes, but we didn't necessarily want to just drop all 36 columns. We first removed essentially all columns that weren't physical characteristics of the object of interest, excepting the label koi_pdisposition. This includes the exoplanet's id and name(s).

**Note:** The columns koi_pdispostion and koi_disposition are very similar. koi_disposition values are "the category of this KOI from the Exoplanet Archive", which includes CONFIRMED planets that have been verified manually by NASA. koi_pdisposition values are "the pipeline flag that designates the most probable physical explanation of the KOI". koi_pdisposition is therefore a better fit for our purposes, because we want to decrease the manual labor of NASA to identify candidate exoplanets and thus want to rely on the physical explanations rather than whether the object of interest is CONFIRMED in the Exoplanet Archive.

After dropping the categorical columns, we still had the NaN problem. There were two columns, koi_teq_err1 and koi_teq_err2 that had no values for any data points, so those were dropped. Then, we reached a spot in our data cleaning process where some decisions had to be made. Most of the continuous columns had err1 and err2 values, which were essentially the confidence interval range for the actual value measured for the column. For example, if koi_depth was 2, err1 could be 1, and err2 could be -0.5, telling us that the actual bounds for koi_depth were between [1.5, 3]. We considered having different columns for upper and lower bounds, but didn't see that the information added by that would be exceedingly helpful. Because of this we chose to essentially ignore all of the error columns and keep the measured values. After doing this, we removed the data points that still contained NaN values. We decided to not impute the missing values due to the variation between the physical characteristics of all of the data points.

Cleaning the data of NaN values and unnecessary columns reduced our dataset to 22 columns and nearly 8000 data points. We are confident that 8000 data points is enough for a successful and generalized classification model.

**Note:** The feature koi_score was not removed during the cleaning process, but we chose to not include it in our final list of features. The definition of this 
feature is "a value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition. The value is calculated from a Monte Carlo technique such that the score's value is equivalent to the frction of iterations where the Robovetter yields a disposition of CANDIDATE." Thus, the koi_score is directly linked to the koi_pdisposition of the data point, our target variable, using complex mathmatical methods. Because of this direct correlation, and the fact that it is not a physical characteristc of the exoplanet, we have chosen to disregard it from consideration for features.

### Feature Selection Methods
#### Visualizing the Pearson Correlation Coefficient Between Pairs of Features
After cleaning the dataset, we were left with 22 features. Since we are focusing on physical characteristics, it is very probable that there are highly correlated variables. We checked this by visualizing the Pearson correlation between each feature. The Pearson correlation coefficient values are between +1 and -1; +1 denotes a strong positive correlation while -1 denotes a strong negative correlation. Using the python visualization library seaborn, we created a heatmap illustrating the Pearson correlation between each variable. 
![](images/pearson.png)
As seen in the Pearson correlation, we have a few semi-correlated features, seen by the darker red and blue colors in the heatmap. However, most of the features are not highly correlated, if at all.

#### Random Forest to Determine Feature Importance
Since we are not considering koi_score (see note above) we are left with 21 features for each data point. It is very likely that most of these features are not necessarily significant for actually predicting the label of the data point. Random Forest is a very popular method to determine feature importance. The model accomplishes this by creating many decision trees, and determining where the splits in those decision trees should optimally be, on which features by maximizing the information gain at each split. By doing this, the Random Forest algorithm can rank the features based on the average information gain of the feature. We plotted the ten most important features found by Random Forest.

![](images/rf_feat_importance.png)

The features and their definitions are in the table below. 

| Feature | Definition  |
|---|---|
| **koi_fpflag_ss** | **Stellar Eclipse Flag**: A KOI that is observed to have a significant secondary event, transit shape, or out-of-eclipse variability, which indicates that the transit-like event is most likely caused by an eclipsing binary. |
| **koi_fpflag_co** | **Centroid Offset Flag**: The source of the signal is from a nearby star, as inferred by measuring the centroid location of the image both in and out of transit, or by the strength of the transit signal in the target's outer (halo) pixels as compared to the transit signal from the pixels in the optimal (or core) aperture. |
| **koi_fpflag_nt** | **Not Transit-Like Flag**: A KOI whose light curve is not consistent with that of a transiting planet. This includes, but is not limited to, instrumental artifacts, non-eclipsing variable stars, and spurious (very low SNR) detections. |
| **koi_fpflag_ec** | **Ephemeris Match Indicates Contamination Flag**: The KOI shares the same period and epoch as another object and is judged to be the result of flux contamination in the aperture or electronic crosstalk.  |
| **koi_prad** | **Planetary Radius (Earth radii)**: The radius of the planet. Planetary radius is the product of the planet star radius ratio and the stellar radius. |
| **koi_depth** | **Transit Depth (parts per million)**: The fraction of stellar flux lost at the minimum of the planetary transit. Transit depths are typically computed from a best-fit model produced by a Mandel-Agol (2002) model fit to a multi-quarter Kepler light curve, assuming a linear orbital ephemeris. |
| **koi_insol** | **Insolation Flux [Earth flux]**: Insolation flux is another way to give the equilibrium temperature. It depends on the stellar parameters (specifically the stellar radius and temperature), and on the semi-major axis of the planet. It's given in units relative to those measured for the Earth from the Sun. |
| **koi_teq** | **Equilibrium Temperature (Kelvin)**: Approximation for the temperature of the planet. The calculation of equilibrium temperature assumes a) thermodynamic equilibrium between the incident stellar flux and the radiated heat from the planet, b) a Bond albedo (the fraction of total power incident upon the planet scattered back into space) of 0.3, c) the planet and star are blackbodies, and d) the heat is evenly distributed between the day and night sides of the planet. |
| **koi_period** | **Orbital Period (days)**: The interval between consecutive planetary transits. |
| **koi_model_snr** | **Transit Signal-to-Noise**: Transit depth normalized by the mean uncertainty in the flux during the transits. |

By just taking the top ten features, we cut our number of features in half. However, five of the ten features have an importance below 0.05, which can generally be classified as "low". However, we cannot know for sure, and our manual cost of trying both the datasets with the top ten features and only the top five features is very low, so we implemented both datasets in the unsupervised learning methods below.  

# Methods
call out features of clusters that indicate candidates (process and describe some math, Why)
## Unsupervised
### K-means
K-means is a clustering algorithm that groups points around a selected K number of centroids. K-means requires a predetermined value of K. The K-means algorithim iteratively assign's data points to the nearest centroid, then moves the centroid assignment to the average of the points assigned to it. This process continues until the algorithim converges. 

Our team tuned our K-means model based on finding the optimal K in an elbow curve graph. The elbow curve graph calculates the sum of square distances from each point and the centroid that the point is assigned to. This calculation is done over a range of K centroids. The range we selected was from 1-15 centroids. The sum of squared distances (or errors) is then ploted on a graph. Next, we ran a function to identify the point of inflection on the curve, known as the "elbow". This value is then used as the value of K for our K-means algorithim.

### Gaussian Mixture Modeling
Gaussian Mixture Modeling results on the exoplanets data still required some data preparation before final results could be interpreted. The primary method the team used to analyze results with GMM was to normalize all non-binary data using StandardScaler(). Once we concatenated all of the team’s data back together, the team then generated a scipy GMM model with the normalized data.
	
From this model the team was able to generate predicted labels and probabilities for different numbers of clusters. The team was also able to visualize the clusters utilizing PCA. Since we cannot visualize more than three dimensions, and only two easily, the team then used the first two PCA vectors for visualizing the different cluster assignments. All results generated by the team are displayed in the results section for GMM.
	
In order to evaluate quality of a model’s fit the team used two techniques. The first method was to compare our predicted labels to the true labels provided by the kepler dataset. This provided a simple standard error percentage we could use to determine the effectiveness of our algorithm. The team opted to not use the sum of the squares ANOVA method because it would prove to yield the same results as our previous method due to the nature of the binary prediction. The second method the team used to determine the best model/best number of clusters was with the AIC & BIC selection criterion. Graphs comparison the AIC/BIC to number of clusters are shown in the results section of GMM. The “best” model was selected on the elbow of the AIC/BIC graph.

### Hierarchical Clustering
One form of unsupervised learning we wanted to conduct on our dataset is hierarchical clustering. This form of clustering, known as bottom-up agglomerative clustering, starts by considering each object as its own cluster. It will then repeatedly join the ‘most similar’ pair of clusters, and will merge it into a parent cluster. This process repeats until there is only one cluster left.

Therefore, a question that arises is, if we have to determine similar clusters, how do we define that similarity? In our implementation, we chose a ‘ward’ linkage, which minimizes the variance of clusters being merged. Like our other unsupervised methods, we standardized our non-binary variables for both a dataset with the top five most important features and top ten most important features. A dendrogram was created to visualize this linkage for both datasets. Then, to determine where to make our ‘cut’ on the dendrogram for each of our datasets, we calculated the silhouette score for clustering from six to 25 clusters (clustering was done using sklearn’s AgglomerativeClustering function), selecting our optimal number of clusters that maximizes the silhouette score. 

A silhouette score measures how close points are to their own clusters but far from other clusters, and a score close to one implies good clustering. Please reference the results section for our outcome in determining the optimal number of clusters for each dataset. Like our other methods, we can then visualize this clustering by plotting the clusters using PCA, and also determine how many candidate planets and false positives reside in each cluster. Results of this analysis can be found in the results section as well. 

### Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a non-parametric, density-based clustering algorithm. DBSCAN detects arbitrarily shaped clusters in the data, where a cluster is defined as a maximal set of density-connected points. A point is described as being a part of a cluster if it is within a core point's &epsilon;-neighborhood, which means the data point is within the distance &epsilon; from the core point. In order to be called a cluster instead of outliers, each cluster must contain at least a minimum number of samples. The two parameters we tuned to create our ideal DBSCAN model were \epsilon and the minimum number of samples for a group of points to be defined as a cluster.


## Supervised
### TODO: Steps Moving Forward

## K-means
### Result
In order to choose the correct number of clusters for the K-means algorithim, the team plotted an elbow curve over both the top 10 and top 5 features. We then ran an elbow identifying function to find the most significant number of clusters to use.

#### Using Top 10 Features

![](images/top10_elbow.png)

**Parameter Values:** K clusters = 4 

| Cluster Number | Number of CANDIDATES | Number of FALSE POSITIVES | **Percent CANDIDATE Objects in Cluster** |
|---|---|---|---|
| 0 | 42 | 1765 | 2% |
| 1 | 0 | 1213 | 0.03% |
| 2 | 0 | 1| 0% |
| 3 | 3987 | 1350 | 75% |

![](images/kmeans_top_10.png)

#### Using Top 5 Features

![](images/top5_elbow.png)

**Parameter Values:** K clusters = 6

| Cluster Number | Number of CANDIDATES | Number of FALSE POSITIVES | **Percent CANDIDATE Objects in Cluster** |
|---|---|---|---|
| 0 | 8 | 495 | 1.6% |
| 1 | 46 | 1129 | 4% |
| 2 | 1| 11 | 8% |
| 3 | 3820 | 1082 | 78% |
| 4 | 153 | 1142 | 2.5% |
| 5 | 1 | 6 | 14% |

![](images/kmeans_top_5_cluster.png)


## Gaussian Mixture Modeling
### Result
  The team found that, using the Gaussian Mixture Model, that 2 clusters was the ideal way to classify potential exoplanets with our chosen features. See below for two GIFs (5/10 feature clustering) of the first two PCA columns being assigned to various clusters for a brief understanding of how our algorithm works:

![](images/5_feature_gmm.gif)

![](images/10_feature_gmm.gif)


However, we used the AIC/BIC criterion to determine the best number of clusters to use. Below is a graph comparing number of clusters to criterion number, with the “elbow/best #” highlighted. Both 5 and 10 features are displayed for reference:

![](images/aic_bic_5.png)

![](images/aic_bic_10.png)

As the above figure highlights, 2 clusters is approximately when the AIC/BIC begins indicating 2 clusters is ideal for the 10 feature model and 3 is appropriate for the 5 feature model. Below is the figure displaying the model with just two clusters. :

![](images/5_feature_final.png

![](images/10_feature_final.png)

## Hierarchical Clustering
### Result
Below is the constructed dendrogram on our scaled dataset using a ward linkage for both our top five most important features and top 10 most important features. Using top five features:

![](images/dendrogram_five.png)

Now, using top 10 features:

![](images/dendrogram_ten.png)

As mentioned briefly in the methods section, we decided to determine the optimal number of clusters by calculating the silhouette score for clustering from six to 25 clusters. Here is a graphical representation of those scores for each of our clusters. First for the five-feature dataset:

![](images/silhouette_scores_five.png)

And the 10-feature dataset:

![](images/silhouette_scores_ten.png)

By locating the maximum silhouette score for each dataset, we determine that our optimal number of clusters is 17 for our top five features with a silhouette score of 0.937 and 11 for our top 10 features with a silhouette score of 0.309. We can then move to reducing our datasets to two principal components using PCA so that we can graphically represent our clustering. Below is the result of our PCA analysis. First is the graph for our top five dataset:

![](images/hier_PCA_five.png)

And top 10:

![](images/hier_PCA_ten.png)

Lastly, we can directly calculate the number of candidate and false positive exoplanets in each of our clusters. Below is the tabular representation of this breakup for the top five features:

| Cluster Number | Number of CANDIDATES | Number of FALSE POSITIVES | **Percent CANDIDATE Objects in Cluster** |
|---|---|---|---|
| 0 | 41 | 1342 | 0.03% |
| 1 | 0 | 596 | 0% |
| 2 | 0 | 49 | 0% |
| 3 | 0 | 458 | 0% |
| 4 | 3987 | 0 | 100% |
| 5 | 0 | 162 | 0% |
| 6 | 1 | 17 | 0.056% |
| 7 | 0 | 105 | 0% |
| 8 | 0 | 313 | 0% |
| 9 | 0 | 5 | 0% |
| 10 | 0 | 312 | 0% |
| 11 | 0 | 366 | 0% |
| 12 | 0 | 113 | 0% |
| 13 | 0 | 53 | 0% |
| 14 | 0 | 72 | 0% |
| 15 | 0 | 1 | 0% |
| 16 | 0 | 1 | 0% |

For the top 10 features dataset:

| Cluster Number | Number of CANDIDATES | Number of FALSE POSITIVES | **Percent CANDIDATE Objects in Cluster** |
|---|---|---|---|
| 0 | 24 | 1735 | 0.014% |
| 1 | 56 | 924 | 0.057% |
| 2 | 3910 | 480 | 0.891% |
| 3 | 10 | 113 | 0.081% |
| 4 | 0 | 170 | 0% |
| 5 | 0 | 1 | 0% |
| 6 | 24 | 218 | 0.099% |
| 7 | 1 | 1 | 0.5% |
| 8 | 0 | 4 | 0% |
| 9 | 1 | 13 | 0.071% |
| 10 | 3 | 306 | 0.010% |

For both datasets, we have a cluster that returns a majority of candidate exoplanets. Regarding the top five dataset, cluster four contains 3987 candidate planets and no false positives, while the top 10 dataset's cluster two contains 3910 candidate planents and only 480 false positives. 

## DBSCAN
### Result
#### Using Top 10 Features
**Parameter Values:** &epsilon; = 1, min_samples = 15
| Cluster Number | Number of CANDIDATES | Number of FALSE POSITIVES | **Percent CANDIDATE Objects in Cluster** |
|---|---|---|---|
| 0 | 3967 | 0 | 100% |
| 1 | 39 | 1213 | 0.03% |
| 2 | 0 | 298 | 0% |
| 3 | 0 | 306 | 0% |
| 4 | 0 | 449 | 0% |
| 5 | 0 | 525 | 0% |
| 6 | 0 | 103 | 0% |
| 7 | 0 | 360 | 0% |
| 8 | 0 | 153 | 0% |
| 9 | 0 | 63 | 0% |
| 10 | 0 | 104 | 0% |
| 11 | 0 | 41 | 0% |

![](images/dbscan_clusters_top10.png)
#### Using Top 5 Features
**Parameter Values:** &epsilon; = 0.5, min_samples = 6

| Cluster Number | Number of CANDIDATES | Number of FALSE POSITIVES | **Percent CANDIDATE Objects in Cluster** |
|---|---|---|---|
| 0 | 3986 | 0 | 100% |
| 1 | 41 | 1377 | 0.03% |
| 2 | 0 | 314 | 0% |
| 3 | 0 | 322 | 0% |
| 4 | 0 | 458 | 0% |
| 5 | 0 | 601 | 0% |
| 6 | 0 | 113 | 0% |
| 7 | 0 | 366 | 0% |
| 8 | 0 | 155 | 0% |
| 9 | 0 | 72 | 0% |
| 10 | 0 | 105 | 0% |
| 11 | 0 | 52 | 0% |

![](images/dbscan_clusters_top5.png)


# Discussion
-(MICAELA)flow chart
![](images/exoplanet.png)
-talk about determining uncertainty 
-we don't yet have error
