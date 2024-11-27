from utils import *
import pandas as pd
import polars as pl
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# Read datasets
consumption = pl.read_parquet("../data/electricity_consumption.parquet")

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