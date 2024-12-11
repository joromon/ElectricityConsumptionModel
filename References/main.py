from utils import *
import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import graphviz

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)
wd_data = "../data"

# Read datasets
consumption = pl.read_parquet(f"{wd_data}/electricity_consumption.parquet")
weather = pl.read_parquet(f"{wd_data}/weather.parquet")
socioeconomic = pl.read_parquet(f"{wd_data}/socioeconomic.parquet")
cadaster = pd.concat([
    gpd.read_file(f"{wd_data}/cadaster_lleida.gml"),
    gpd.read_file(f"{wd_data}/cadaster_alcarras.gml"),
    gpd.read_file(f"{wd_data}/cadaster_alpicat.gml")])
postalcodes = gpd.read_file(f"{wd_data}/postal_codes_lleida.gpkg")

# Join cadaster and postalcode dataframes
postalcodes.rename({"PROV":"provincecode", "CODPOS":"postalcode"}, axis=1, inplace=True)
cadaster["centroid"] = cadaster.geometry.centroid
postalcodes = postalcodes.to_crs(cadaster.crs)
cadaster = gpd.sjoin(
    cadaster,  # GeoDataFrame with building centroids
    postalcodes[['postalcode', 'geometry']],  # Postal code polygons
    how='left',  # Include all rows from cadaster_df
    predicate='within'  # Match if the centroid is within a postal code polygon
)
cadaster = cadaster[
    ["postalcode", "currentUse", "conditionOfConstruction",
     "numberOfDwellings", "reference", "value"]]
cadaster.rename({
    "numberOfDwellings": "households",
    "reference": "cadastralref",
    "value": "builtarea",
    "currentUse": "currentuse",
    "conditionOfConstruction": "condition"}, axis=1, inplace=True)
cadaster = cadaster[~pd.isna(cadaster.postalcode) &
                    (cadaster.currentuse == "1_residential") &
                    (cadaster.condition == "functional")]
cadaster = pl.DataFrame(
    cadaster.groupby("postalcode")[["builtarea","households"]].sum().reset_index())

# Data transformation
## Calculate the local time, hour and date of each data point
consumption = consumption.with_columns(
    pl.col("time").dt.convert_time_zone("Europe/Madrid").alias("localtime"))
consumption = consumption.with_columns(
    pl.col("localtime").dt.hour().alias("hour"),
    pl.col("localtime").dt.date().alias("date")
)
consumption = consumption.sort([pl.col("postalcode"), pl.col("localtime")])

## Group by 'postalCode' and calculate rolling mean, std, and z-norm
rolling_results = consumption.group_by("postalcode", maintain_order=True).agg([
    pl.col("consumption").rolling_mean(48,min_periods=4,center=True).alias("rolling_mean"),
    pl.col("consumption").rolling_std(48,min_periods=4,center=True).alias("rolling_std"),
    pl.col("consumption").rolling_quantile(quantile=0.1,window_size=168,min_periods=24,center=True,interpolation="nearest").alias("rolling_q10")
])
consumption = pl.concat([
    consumption,
    rolling_results.explode(["rolling_mean","rolling_std","rolling_q10"]).
    select(pl.all().exclude("postalcode"))],
    how="horizontal"
)
consumption = consumption.with_columns(
    ((pl.col("consumption") - pl.col("rolling_mean")) / pl.col("rolling_std")).alias("z_norm")
)

## Plot z-normalisation values
consumption_plotter(
        consumption.filter(pl.col("postalcode")=="25001"),
        html_file=f"plots/exemple_z_norm.html",
        y_columns=["z_norm"],
        y_title="znorm")

## Filter each data point with z_norm < z_norm_threshold
max_threshold = 4
min_threshold = -2
consumption = consumption.with_columns(
    pl.when((pl.col("z_norm") < max_threshold) &
            (pl.col("z_norm") > min_threshold) &
            (pl.col("consumption") > (pl.col("rolling_q10")*0.7))).
    then(pl.col("consumption")).
    otherwise(np.nan).alias("consumption_filtered")
)

## Plot consumption vs. consumption_filtered
for postalcode in consumption["postalcode"].unique():
    consumption_plotter(
        consumption.filter(pl.col("postalcode")==postalcode),
        html_file=f"plots/input_consumption_{postalcode}.html",
        y_columns=["consumption", "consumption_filtered"],
        y_title="kWh")
    plot_daily_load_curves_with_centroids_to_pdf(
        df=consumption.filter(pl.col("postalcode")==postalcode),
        pdf_path=f"plots/daily_load_curves_all_{postalcode}.pdf")

####
# TASK1: CLUSTERING DAILY LOAD CURVES
####

# Preparing the dataset for clustering
## Select the number of hours to aggregate the different parts of day
## n_hours = 1 would not aggregate the original hourly time series
n_hours = 3
## Intraday percentage of consumption
consumption_long = (consumption.join(
        consumption.group_by(["postalcode","date"]).agg(
            (pl.col("consumption_filtered").mean()*24).alias("daily_consumption")
        ),
        on=["postalcode","date"]).
    with_columns(
        (((pl.col("localtime").dt.hour()/n_hours).floor()) * n_hours).alias("hour")
    ).
    group_by(["postalcode","hour","date","daily_consumption"]).agg(
        pl.col("consumption_filtered").mean() * n_hours
    ).
    with_columns(
        (pl.col("consumption_filtered")*100 / pl.col("daily_consumption")).alias("consumption")
    )
)
## Intraday absolute consumption
# consumption_long = consumption.with_columns(
#         (((pl.col("localtime").dt.hour()/n_hours).floor())*n_hours).alias("hour")
#     ).group_by(["postalcode","hour","date"]).agg(
#         pl.col("consumption_filtered").mean() * n_hours
#     )
## Transform the dataset to wide
consumption_wide = (consumption_long.
                    sort(["postalcode","hour","date"]).
                    select(["postalcode","date", "hour", "consumption"]).
                    pivot(index=["postalcode","date"], on="hour"))

consumption_wide = consumption_wide.to_pandas()
consumption_wide.set_index(["postalcode","date"],inplace=True)

# Scale the data
consumption_wide_min_max = pd.DataFrame(
    MinMaxScaler().fit_transform(consumption_wide),
    columns=consumption_wide.columns,
    index=consumption_wide.index)
consumption_wide_znorm = pd.DataFrame(
    StandardScaler().fit_transform(consumption_wide),
    columns=consumption_wide.columns,
    index=consumption_wide.index)

# Create a scaled object
consumption_wide_scaled = [
    (consumption_wide, "NoScaling"),  # "Without scaling
    (consumption_wide_min_max, "MinMaxScaling"),  # Min-Max scaling
    (consumption_wide_znorm, "ZnormScaling"),  # Standard scaling
]

# Agglomerative hierarchical clustering pipeline
range_n_clusters = list(range(2, 10))

for X, scaling_type in consumption_wide_scaled:
    silhouette_scores = []  # empty list to store silhouette scores
    clustering_X = X.dropna(axis=0)
    index_X = clustering_X.index.to_frame(index=False)
    clustering_X = clustering_X.reset_index(drop=True)

    for n_clusters in range_n_clusters:
        print(f"Agglomerative Hierarchical Clustering {scaling_type} data with {n_clusters} clusters")
        alg = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True, linkage="ward")  # linkage="ward", "complete", "average", "single")
        cl_results = alg.fit(clustering_X)
        cluster_labels = cl_results.labels_
        silhouette_avg = silhouette_score(
            clustering_X,
            cluster_labels)
        silhouette_scores.append(silhouette_avg)

        # Plot the dendogram
        plt.figure(figsize=(15, 8))
        plt.title(f"Hierarchical Clustering Dendrogram - {scaling_type}")
        # plot the top three levels of the dendrogram
        plot_dendrogram(cl_results, truncate_mode='lastp', p=n_clusters)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.ylabel("Distance based on linkage")
        plt.savefig(f"plots/hierarchical_clustering_dendrogram_{n_clusters}_{scaling_type}.pdf", format="pdf")

        # Plot the daily load curves
        plot_daily_load_curves_with_centroids_to_pdf(
            df=(consumption.select(pl.all().exclude("cluster")).
                join(
                    consumption.group_by(["postalcode", "date"]).agg(
                        (pl.col("consumption_filtered").mean() * 24).alias("daily_consumption")
                    ),
                    on=["postalcode", "date"]).
                with_columns(
                    (pl.col("consumption_filtered") * 100 / pl.col("daily_consumption")).alias("consumption_filtered")
                ).
                join(
                    pl.DataFrame(
                        pd.concat([
                            index_X.reset_index(drop=True),  # Ensure no index issues
                            pd.DataFrame(cluster_labels, columns=["cluster"])],
                            axis=1
                        )).
                    with_columns(pl.col("date").cast(pl.Date)),
                on=["postalcode","date"])),
            pdf_path=f"plots/daily_load_curves_n_clusters_{n_clusters}_{scaling_type}.pdf",
            add_in_title = scaling_type)

    # Plot silhouette results
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title(f"Optimisation of the number of clusters - {scaling_type}")
    plt.savefig(f"plots/hierarchical_clustering_silhouette_{scaling_type}.pdf", format="pdf")

# Select the clustering characteristics based on analysis of results
alg = AgglomerativeClustering(n_clusters=3, compute_distances=True, linkage="ward")  # linkage="ward", "complete", "average", "single")
clustering_X = consumption_wide_scaled[1][0].dropna(axis=0)
index_X = clustering_X.index.to_frame(index=False)
clustering_X = clustering_X.reset_index(drop=True)
cl_results = alg.fit_predict(clustering_X)
clustering_results = (pl.concat([
        pl.DataFrame(index_X),
        pl.DataFrame({"cluster": cl_results})],
        how="horizontal").
    with_columns(
    pl.col("date").cast(pl.Date)))

####
# TASK2: CLASSIFICATION MODEL FOR DAY AHEAD PREDICTING DAILY LOAD CURVES
####

# Aggregating the weather data to daily
weather_daily = (weather.
    with_columns(
        pl.col("time").dt.convert_time_zone("Europe/Madrid").alias("localtime")).
    with_columns(
        pl.col("localtime").dt.date().alias("date")).
    group_by(["date","postalcode"]).
    agg(
        (pl.col("airtemperature").drop_nans().mean()).round(2).
            alias("airtemperature"),
        (pl.col("relativehumidity").drop_nans().mean()).round(2).
            alias("relativehumidity"),
        (pl.col("totalprecipitation").drop_nans().mean()*24).round(2).
            alias("totalprecipitation"),
        (pl.col("ghi").drop_nans().mean()).round(2).
            alias("ghi"),
        (pl.col("sunelevation").drop_nans().mean()).round(2).
            alias("sunelevation")
    )
)

# Aggregate to daily consumption, including the clustering of daily load curves
consumption_daily = (consumption.
    group_by(["date","postalcode"]).
    agg(
        (pl.col("consumption_filtered").drop_nans().mean()*24).round(2).
        alias("consumption")).
    join(
        clustering_results, on = ["postalcode", "date"]))

# Preparing the dataset used by the classification model
all_data_daily = (consumption_daily.
    with_columns(
        pl.col("date").dt.year().cast(pl.Int64).alias("year")).
    join(
        socioeconomic, on = ["postalcode", "year"]).
    join(
        cadaster, on=["postalcode"]).
    join(
        weather_daily, on=["postalcode", "date"]).
    with_columns(
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday")).
    sort(
        ["postalcode", "date"]).
    to_pandas()
)

# Decision tree for classification
X_train, X_test, y_train, y_test = train_test_split(
    all_data_daily.drop(["date", "postalcode", "cluster", "consumption"], axis=1),
    all_data_daily.cluster,
    random_state=42,
    test_size=0.2
)
clf = DecisionTreeClassifier(
    # **** Hyper-parameters tuning ***
    criterion="entropy", # "gini", # entropy
    # ** Pruning **
    max_depth=4,
    min_samples_leaf=20,
    random_state=42)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
print(f"Model Training Accuracy: {round(accuracy*100,2)}%")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Testing Accuracy: {round(accuracy*100,2)}%")

# Visualise the decision tree
dot_data = tree.export_graphviz(
    clf, out_file=None,
    feature_names=X_train.columns, filled=True, rounded=True, special_characters=True
)
graph = graphviz.Source(dot_data)
graph.format = 'png'  # specify the output format
graph.render(filename='plots/decision_tree_classifier_graph', directory='.', cleanup=True)


####
# TASK3: HOURLY ELECTRICITY REGRESSION MODEL
####

all_data_hourly = (consumption.
    select(['postalcode', 'localtime', 'hour', 'contracts', 'consumption_filtered', 'consumption']).
    with_columns(
        pl.col("localtime").dt.year().cast(pl.Int64).alias("year")).
    join(
        socioeconomic, on = ["postalcode", "year"]).
    join(
        cadaster, on=["postalcode"]).
    join(
        weather.
        with_columns(
            pl.col("time").dt.convert_time_zone("Europe/Madrid").
                alias("localtime")),
        on=["postalcode", "localtime"]).
    with_columns(
        pl.col("localtime").dt.month().cast(pl.Utf8).alias("month"),
        pl.col("localtime").dt.weekday().cast(pl.Utf8).alias("weekday")).
    sort(
        ["localtime"]).
    to_pandas()
)

# Define features and target
X = (all_data_hourly.
    drop(
        ['postalcode', 'localtime', 'time', 'consumption_filtered', 'consumption'],
        axis=1))
X = X[~pd.isna(all_data_hourly.consumption_filtered)]
y = all_data_hourly['consumption_filtered']
y = y[~pd.isna(all_data_hourly.consumption_filtered)]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Preprocessing for numerical data: scaling
numerical_transformer = StandardScaler()

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

models = {
    "Decision Tree": DecisionTreeRegressor(min_samples_leaf=10, max_depth=20),
    "Decision Tree 2": DecisionTreeRegressor(min_samples_leaf=20, max_depth=6),
    "Linear Regression": LinearRegression()
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate each model
pipelines = {}
for model_name, model in models.items():
    pipelines[model_name] = evaluate_model(model_name, preprocessor, model,
                                           X_train, y_train,
                                           X_test, y_test)


