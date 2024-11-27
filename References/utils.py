import plotly.express as px
import plotly.io as pio
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colormaps

def plot_daily_load_curves_with_centroids_to_pdf(df: pl.DataFrame, pdf_path: str):
    """
    Plots daily load curves with centroids for each cluster from a Polars DataFrame and saves the plot to a PDF.

    Parameters:
        df (pl.DataFrame): A Polars DataFrame containing 'date', 'hour', 'cluster', and 'consumption' columns.
        pdf_path (str): The file path where the PDF should be saved.
    """
    # Ensure the 'cluster' column exists; if not, create it filled with 1's
    if "cluster" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("cluster"))

    # Ensure the DataFrame is sorted by date and hour
    df = df.sort(["date", "hour"])

    # Group by 'date' and 'cluster', and collect consumption data for each day
    daily_curves = df.group_by(["date", "cluster"]).agg(pl.col("consumption_filtered"))

    # Create a centroid DataFrame by averaging consumption for each hour and cluster
    centroids = (
        df.group_by(["hour", "cluster"]).
        agg(pl.col("consumption_filtered").drop_nans().mean().alias("consumption_filtered")).
        sort(["hour","cluster"]).
        pivot(index="cluster", on="hour")
    )

    # Convert to pandas for easy plotting with matplotlib
    daily_curves_pandas = daily_curves.to_pandas()
    centroids_pandas = centroids.to_pandas()

    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(10, 6))
        unique_clusters = df["cluster"].unique().to_list()
        colors = colormaps["tab10"]  # Use the updated way to access colormaps
        for _, row in daily_curves_pandas.iterrows():
            date = row["date"]
            cluster = row["cluster"]
            consumption = row["consumption_filtered"]
            # Plot individual daily load curves with low opacity
            plt.plot(range(len(consumption)), consumption, alpha=0.1, color=colors(cluster / len(unique_clusters)),
                     label=None)
        # Plot centroids with bold lines
        for cluster in unique_clusters:
            centroid = centroids_pandas[centroids_pandas["cluster"] == cluster].drop("cluster", axis=1).iloc[0].to_numpy()
            plt.plot(range(len(centroid)), centroid, linewidth=2.5, label=f"Cluster {cluster}",
                     color=colors(cluster / len(unique_clusters)))
        # Add labels and legend
        plt.xlabel("Hour of Day")
        plt.ylabel("Consumption")
        plt.title("Daily Load Curves with Cluster Centroids")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()

def consumption_plotter(consumption, html_file, y_columns, y_title):

    consumption = consumption.sort('time')

    # Convert the Polars DataFrame to Pandas for compatibility with Plotly
    consumption_df = consumption.to_pandas()

    # Create a separate interactive line plot for each postal code with Plotly
    figs = []
    postal_codes = consumption_df['postalcode'].unique()

    for postal_code in postal_codes:
        subset = consumption_df[consumption_df['postalcode'] == postal_code]

        fig = px.line(
            subset,
            x='time',
            y=y_columns,
            title=f'Postal Code: {postal_code}',
            labels={'value': 'Value', 'variable': 'Metric'},
        )
        fig.update_layout(xaxis_title='Time', yaxis=dict(
            tickformat=".2f",  # Ensure numbers are displayed with two decimal places
            title=y_title,   # Y-axis title
        ))
        figs.append(fig)

    # Save each plot to an HTML file and show it
    with open(html_file, "w") as f:
        for fig in figs:
            f.write(pio.to_html(fig, full_html=False))


def weather_plotter(weather, html_file):
    # Assuming 'weather' is a Polars DataFrame and needs to be sorted by time
    weather = weather.sort('time')

    # Convert the sorted Polars DataFrame to Pandas for compatibility with Plotly
    weather_df = weather.to_pandas()

    # Create a separate interactive line plot for each postal code with Plotly
    figs = []
    postal_codes = weather_df['postalcode'].unique()

    for postal_code in postal_codes:
        subset = weather_df[weather_df['postalcode'] == postal_code]
        fig = px.line(
            subset,
            x='time',
            y=['airtemperature'],
            title=f'Postal Code: {postal_code}',
            labels={'airtemperature': 'Air Temperature'},
        )
        fig.update_layout(xaxis_title='Time', yaxis_title='Air Temperature')
        figs.append(fig)

    # Save each plot to an HTML file and show it
    with open(html_file, "w") as f:
        for fig in figs:
            f.write(pio.to_html(fig, full_html=False))