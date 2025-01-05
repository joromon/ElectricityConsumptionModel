import random
import pandas as pd
import polars as pl
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_regression_model(model_name, preprocessor, model, hyperparams_ranges, cross_val,
                              X_train, y_train, X_test, y_test):
    # Create and evaluate the pipeline with the given model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=hyperparams_ranges,
                               cv=cross_val, n_jobs=-1, verbose=1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_  # Get the best estimator from the grid search
    y_pred = best_pipeline.predict(X_test)

    best_params = grid_search.best_params_

    # Calculate Mean Squared Error (MSE) and Root MSE (RMSE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    ## Coefficient of Variation of the RMSE
    cvrmse = rmse / y_test.mean()

    print(f"Best parameters: {best_params}")
    print(f"{model_name} - Root Mean Squared Error: {round(rmse, 2)}")
    print(f"{model_name} - Coefficient of Variation of the Root Mean Squared Error: {round(cvrmse * 100, 2)} %")

    return best_pipeline


def plot_regression_results(pipeline, df, filename, postal_code, model_name, hours=96, npred=1):
    """
    Plot and save actual vs predicted consumption for a specific postal code into a single PDF.

    Parameters:
        pipeline: Trained model pipeline with a `.predict` method.
        df (DataFrame): Data containing postal codes, 'localtime', and 'consumption'.
        filename (str): The file name to store the plots.
        postal_code (str/int): The postal code to filter data.
        model_name (str): The name of the model used for predictions.
        hours (int): Number of hours to plot.
        npred (int): Number of random plots to generate.
    """
    # Filter data by postal code
    postal_filter = df['postalcode'] == postal_code
    filtered_df = df[postal_filter].copy()

    # Drop irrelevant columns for prediction
    X_test_filtered = filtered_df.drop(["localtime", "consumption"], axis=1)

    # Predict consumption and add it to the dataframe
    y_pred = pipeline.predict(X_test_filtered)
    filtered_df["predicted"] = y_pred

    # Create a single PDF file to store all plots
    with PdfPages(filename) as pdf:
        for i in range(npred):
            # Randomly select a continuous range of data
            max_start = len(filtered_df) - hours
            if max_start <= 0:
                print("Not enough data to plot the specified number of hours.")
                return

            rand_start = random.randint(0, max_start)
            df_slice = filtered_df.iloc[rand_start:rand_start + hours]

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(df_slice["localtime"], df_slice["predicted"], label='Predicted', marker='x', linestyle='--',
                     markersize=3)
            plt.plot(df_slice["localtime"], df_slice["consumption"], label='Actual', marker='o', linestyle='-',
                     markersize=3)

            plt.title(f"Actual vs Predicted Consumption\nPostal Code: {postal_code} ({model_name})")
            plt.xlabel("Time")
            plt.ylabel("Consumption")
            plt.legend()
            plt.grid(True)

            # Format x-axis for date display
            ax = plt.gca()  # Get current axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Add the current figure to the PDF
            pdf.savefig()
            plt.close()

        # Add metadata to the PDF
        d = pdf.infodict()
        d['Title'] = f"Prediction Results for Postal Code {postal_code} ({model_name})"
        d['Author'] = 'Your Name'
        d['Subject'] = 'Actual vs Predicted Consumption Comparison'
        d['Keywords'] = 'Consumption, Prediction, Regression, Model'
        d['CreationDate'] = pd.Timestamp.now()



def prepare_hourly_data(consumption, weather, socioeconomic, cadaster):
    """Prepare the hourly data by joining all necessary datasets."""
    
    consumption = pl.from_pandas(consumption)
    weather = pl.from_pandas(weather)
    socioeconomic = pl.from_pandas(socioeconomic)
    cadaster = pl.from_pandas(cadaster)

    all_data_hourly = (consumption
        .select(['postalcode', 'localtime', 'hour', 'contracts', 'consumption_filtered', 'consumption'])
        .with_columns(pl.col("localtime").dt.year().cast(pl.Int64).alias("year"))
        .join(socioeconomic, on=["postalcode", "year"])
        .join(cadaster, on=["postalcode"])
        .join(weather.with_columns(pl.col("time").dt.convert_time_zone("Europe/Madrid").alias("localtime")), 
              on=["postalcode", "localtime"])
        .with_columns(
            pl.col("localtime").dt.month().cast(pl.Utf8).alias("month"),
            pl.col("localtime").dt.weekday().cast(pl.Utf8).alias("weekday"))
        .sort(["localtime"])
        .to_pandas()
    )
    return all_data_hourly

def preprocess_features_and_target(all_data_hourly):
    """Preprocess the features (X) and target (y) from the hourly data."""
    X = all_data_hourly.drop(['localtime', 'time', 'consumption_filtered', 'consumption'], axis=1)
    X = X[~pd.isna(all_data_hourly['consumption_filtered'])]
    y = all_data_hourly['consumption_filtered']
    y = y[~pd.isna(all_data_hourly['consumption_filtered'])]
    return X, y

def preprocess_data(X):
    """Preprocess features (scaling for numerical and one-hot encoding for categorical)."""
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
        ]
    )
    return preprocessor

def split_data(X, y, postalcodes):
    """Split data into training and testing sets, considering time series split."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(20 * 96 * len(postalcodes)) / len(X), shuffle=False)
    return X_train, X_test, y_train, y_test

def create_and_train_model(preprocessor, X_train, y_train, X_test, y_test, hyperparams_ranges):
    """Create a Decision Tree model, apply the preprocessor and train the model."""
    models = {
        "Decision Tree": DecisionTreeRegressor()
    }
    
    # Initialize pipeline and evaluate the model
    pipelines = {}
    for model_name, model in models.items():
        pipelines[model_name] = evaluate_regression_model(
        model_name=model_name,
        preprocessor=preprocessor,
        model=model,
        hyperparams_ranges=hyperparams_ranges[model_name],
        cross_val=TimeSeriesSplit(n_splits=5, test_size=96),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    return pipelines

def visualize_model_results(pipelines, all_data_hourly, postalcodes):
    """Visualize regression results for certain postal codes."""
    for model_name, pipeline in pipelines.items():
        for postal_code in postalcodes:
            plot_regression_results(
                pipeline=pipeline,
                df=all_data_hourly.drop(['time', 'consumption_filtered'], axis=1),
                filename=f"plots/results_{model_name}_postalcode_{postal_code}.pdf",
                postal_code=postal_code,
                model_name=model_name,
                hours=96,
                npred=10
            )

def electricity_consumption_forecast(consumption, weather, socioeconomic, cadaster, postalcodes):
    """Main function to forecast electricity consumption."""
    
    print("\n### ELECTRICITY CONSUMPTION FORECAST ###")

    # Prepare data
    print("Preparing data...")
    all_data_hourly = prepare_hourly_data(consumption, weather, socioeconomic, cadaster)

    # Preprocess features and target
    print("Preprocessing features and target...")
    X, y = preprocess_features_and_target(all_data_hourly)

    # Preprocess the data (scaling and encoding)
    print("Preprocessing data...")
    preprocessor = preprocess_data(X)

    # Split the data into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, postalcodes)

    # Create and train the model
    print("Training model...")
    hyperparams_ranges = {
        "Decision Tree": {
            'model__max_depth': [15, 25, 40, 60],
            'model__min_samples_leaf': [5, 10, 15, 20],
        }
    }
    pipelines = create_and_train_model(preprocessor, X_train, y_train, X_test, y_test, hyperparams_ranges)

    # Visualize the results
    print("Visualizing results...")
    visualize_model_results(pipelines, all_data_hourly, postalcodes)

    print("Electricity consumption forecasting completed.")
