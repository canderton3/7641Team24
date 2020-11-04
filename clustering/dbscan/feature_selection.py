import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sb

# Load planet data
def load_data():
    exoplanet_data = pd.read_csv(r"../../data/exoplanet.csv", sep=',')
    print("COLUMNS WITH NANS: {}, TOTAL NUMBER: {}".format(exoplanet_data.columns[exoplanet_data.isna().any()].tolist(), len(exoplanet_data.columns[exoplanet_data.isna().any()].tolist())))
    exoplanet_data.drop('rowid', axis=1, inplace=True)
    exoplanet_data.drop('kepid', axis=1, inplace=True)
    exoplanet_data.drop('kepoi_name', axis=1, inplace=True)
    exoplanet_data.drop('kepler_name', axis=1, inplace=True)
    exoplanet_data.drop('koi_disposition', axis=1, inplace=True)
    exoplanet_data.drop('koi_tce_delivname', axis=1, inplace=True)
    exoplanet_data.drop('koi_teq_err1', axis=1, inplace=True)
    exoplanet_data.drop('koi_teq_err2', axis=1, inplace=True)

    # Drop Err columns
    exoplanet_data = exoplanet_data[exoplanet_data.columns.drop(list(exoplanet_data.filter(regex='err')))]

    # Find columns with NaNs
    print("COLUMNS WITH NANS: {}, TOTAL NUMBER: {}".format(exoplanet_data.columns[exoplanet_data.isna().any()].tolist(), len(exoplanet_data.columns[exoplanet_data.isna().any()].tolist())))
    exoplanet_data.dropna(axis=0, inplace=True)

    # Size after cleaning
    print(exoplanet_data.columns.to_list())
    print(len(exoplanet_data.columns.to_list()))
    print(len(exoplanet_data))
    return exoplanet_data

# Plot Pearson correlation with all variables
def pearson_corr(df):
    pearson_df = df.corr(method="pearson")
    ax = plt.axes()
    pearson_heatmap = sb.heatmap(pearson_df, 
            xticklabels=pearson_df.columns,
            yticklabels=pearson_df.columns,
            cmap='RdBu_r',
            linewidth=0.5,
            ax = ax)
    ax.set_title('Pearson Correlation Heatmap')
    plt.show()
    '''
    #Correlation with output variable
    target = abs(pearson_df["koi_score"])
    #Selecting highly correlated features
    relevant_features = target[target>0.3]
    print("Extra Relevant Features: {}".format(relevant_features))
    '''

# Select K Best
def select_k_best(df):
    target = df['koi_pdisposition']
    feats = df.drop('koi_pdisposition', axis=1)

    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(feats,target)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(feats.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Features','Score']  #naming the dataframe columns
    print(featureScores.nlargest(20,'Score'))  #print 10 best features

# Check Feature Importance
def feat_importance(df):
    target = df['koi_pdisposition']
    feats = df.drop('koi_pdisposition', axis=1)

    model = ExtraTreesClassifier()
    model.fit(feats,target)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=feats.columns)
    feat_importances.nlargest(10).plot(kind='barh').get_figure().savefig("feature_importance.png")

def rf(df):
    target = df['koi_pdisposition']
    feats = df.drop('koi_pdisposition', axis=1)
    feats = feats.drop('koi_score', axis=1)
    rf_clf = RandomForestClassifier()
    rf_clf.fit(feats, target)
    feature_importance = rf_clf.feature_importances_
    most_important_features = feats.columns[np.argsort(rf_clf.feature_importances_)[::-1]]
    feat_importances = pd.Series(feature_importance, index=feats.columns)

    ax = plt.axes()
    rf_importance_plt = feat_importances.nlargest(10).plot(kind='barh', ax = ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Feature Importance")
    plt.show()
    print("RF IMPORTANT FEATURES: ", feature_importance)
    print("MOST IMPORTANT FEATURES: ", most_important_features)

# Output data with features selected
if __name__ == "__main__":
    df = load_data()
    print(df.head(20))
    print(df.shape)
    # Look at pearson correlation
    #pearson_corr(df)
    # Random Forest Feature selection
    rf(df)
    # Most important: koi_score, koi_fpflag_ss, koi_fpflag_nt, koi_fpflag_ec, koi_fpflag_co, koi_depth, koi_period, koi_teq, koi_model_snr 
    # Output CSV with most important features
    top_5 = df[["koi_pdisposition", "koi_fpflag_co", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_ec", "koi_prad"]]
    top_10 = df[["koi_pdisposition", "koi_fpflag_co", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_ec", "koi_prad", "koi_depth", "koi_teq", "koi_impact", "koi_insol", "koi_model_snr"]]

    top_5.to_csv("../../data/exoplanet_cleanedrf_top_5.csv", index=False)
    top_10.to_csv("../../data/exoplanet_cleanedrf_top_10.csv", index=False)