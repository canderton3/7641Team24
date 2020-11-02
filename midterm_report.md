# Introduction/Background
-(MICAELA) pull from proposal

# Problem definition
-(MICAELA) pull from proposal

# Data Collection
-(MICAELA) pull from proposal

## Exoplanet Dataset
The dataset was sourced from Kaggle, linked [here](https://www.kaggle.com/nasa/kepler-exoplanet-search-results), sourced directly from NASA. The inital size was 3.52 MB, with 50 columns and around 10,000 data points. Each data point contains information about physical characteristics of already classified objects identified by Kepler; the data point also has a label (koi_pdisposition) designating it as either a CANDIDATE or a FALSE POSITVE. Our goal is to identify CANDIDATE planets.

Thankfully the dataset came with a thorough data dictionary, linked [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html). Our goal was to use the possible exoplanet's physical characteristics to identify the object of interest as either a candidate or false positive, so we we focused mostly on the physical characteristics columns. More information about feature selection is below.

## Data Cleaning and Preparation
### Cleaning
Before dropping any columns, 36 columns contained at least some NaN values. This obviously was not going to be helpful for our purposes, but we didn't necessarily want to just drop all 36 columns. We first removed essentially all columns that weren't physical characteristics of the object of interest, excepting the label koi_pdisposition. This includes the exoplanet's id and name(s).

**Note:** The columns koi_pdispostion and koi_disposition are very similar. koi_disposition values are "the category of this KOI from the Exoplanet Archive", which includes CONFIRMED planets that have been verified manually by NASA. koi_pdisposition values are "the pipeline flag that designates the most probable physical explanation of the KOI". koi_pdisposition is therefore a better fit for our purposes, because we want to decrease the manual labor of NASA to identify candidate exoplanets and thus want to rely on the physical explanations rather than whether the object of interest is CONFIRMED in the Exoplanet Archive.

After dropping the categorical columns, we still had the NaN problem.

### Feature Selection Methods


# Methods
call out features of clusters that indicate candidates (process and describe some math, Why)
## Unsupervised
### K-means
### Gaussian Mixture Modeling
### Hierarchical Clustering
### DBSCAN

## Supervised
### TODO: Steps Moving Forward

# Results
--silhouette coefficient 
--number of cluster
--number of candidate / false positive exoplanets in each cluster
## K-means
### Result
## Gaussian Mixture Modeling
### Result
## Hierarchical Clustering
### Result
## DBSCAN
### Result


# Discussion
-(MICAELA)flow chart
-talk about determining uncertainty 
-we don't yet have error
