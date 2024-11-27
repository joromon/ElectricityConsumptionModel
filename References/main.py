import polars as pl
from utils import *
import numpy as np
import os

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
window_size = 24
rolling_results = consumption.group_by("postalcode", maintain_order=True).agg([
    pl.col("consumption").rolling_mean(window_size,min_periods=int(window_size/8),center=True).alias("rolling_mean"),
    pl.col("consumption").rolling_std(window_size,min_periods=int(window_size/8),center=True).alias("rolling_std")
])
consumption = pl.concat([
    consumption,
    rolling_results.explode(["rolling_mean","rolling_std"]).select(pl.all().exclude("postalcode"))],
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
            (pl.col("consumption")>1)).
    then(pl.col("consumption")).
    otherwise(np.nan).alias("consumption_filtered")
)

## Plot consumption vs. consumption_filtered
os.makedirs("plots", exist_ok=True)
for postalcode in consumption["postalcode"].unique():
    consumption_plotter(
        consumption.filter(pl.col("postalcode")==postalcode),
        html_file=f"plots/input_consumption_{postalcode}.html",
        y_columns=["consumption", "consumption_filtered"],
        y_title="kWh")
    plot_daily_load_curves_with_centroids_to_pdf(
        consumption.filter(pl.col("postalcode")==postalcode),
        pdf_path=f"plots/daily_load_curves_all_{postalcode}.pdf")

