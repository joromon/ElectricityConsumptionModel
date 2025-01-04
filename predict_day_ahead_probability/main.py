import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
import sys


def preprocess_weather_data(weather):
    """
    Aggregate weather data to daily resolution.
    """
    weather["time"] = pd.to_datetime(weather["time"])
    weather["localtime"] = weather["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid")
    weather["date"] = weather["localtime"].dt.date
    weather_daily = weather.groupby(["date", "postalcode"]).agg({
        "airtemperature": "mean",
        "relativehumidity": "mean",
        "totalprecipitation": lambda x: x.sum() * 24,
        "ghi": "mean",
        "sunelevation": "mean"
    }).round(2).reset_index()
    
    return weather_daily

def preprocess_consumption_data(consumption, clustering_results):
    """
    Aggregate consumption data to daily resolution and merge clustering results.
    """
    if isinstance(clustering_results, pl.DataFrame):
        clustering_results = clustering_results.to_pandas()

    clustering_results["date"] = pd.to_datetime(clustering_results["date"]).dt.date
    consumption["date"] = pd.to_datetime(consumption["date"]).dt.date
    consumption_daily = consumption.groupby(["date", "postalcode"]).agg({
        "consumption_filtered": lambda x: x.mean() * 24
    }).round(2).rename(columns={"consumption_filtered": "consumption"}).reset_index()
    consumption_daily = consumption_daily.merge(clustering_results, on=["postalcode", "date"], how="left")
    return consumption_daily

def prepare_classification_dataset(
        consumption_daily,
        weather_daily,
        socioeconomic,
        cadaster
    ):
    """
    Merge and prepare all data for classification.
    """
    socioeconomic["year"] = pd.to_datetime(socioeconomic["year"]).dt.year
    consumption_daily["year"] = pd.to_datetime(consumption_daily["date"]).dt.year
    all_data = consumption_daily.merge(socioeconomic, on=["postalcode", "year"], how="left")
    all_data = all_data.merge(cadaster, on="postalcode", how="left")
    all_data = all_data.merge(weather_daily, on=["postalcode", "date"], how="left")
    all_data["month"] = pd.to_datetime(all_data["date"]).dt.month
    all_data["weekday"] = pd.to_datetime(all_data["date"]).dt.weekday
    return all_data

def train_decision_tree_classifier(X, y):
    """Train a Decision Tree Classifier and evaluate its performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        min_samples_leaf=20,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Model Training Accuracy: {round(training_accuracy * 100, 2)}%")

    y_test_pred = clf.predict(X_test)
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Model Testing Accuracy: {round(testing_accuracy * 100, 2)}%")

    return clf, X_train.columns

def visualize_decision_tree(clf, feature_names, output_path="plots/decision_tree_classifier_graph"):
    """Visualize and save the Decision Tree."""
    dot_data = tree.export_graphviz(
        clf, out_file=None,
        feature_names=feature_names,
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render(filename=output_path, directory='.', cleanup=True)

def predict_day_ahead_probability(
        consumption,
        weather,
        socioeconomic,
        cadaster,
        clustering_results
    ):

    """Main function to predict day-ahead load curve probabilities."""

    print("\n### PREDITCT DAY AHEAD PROBABILITY ###")


    print("Preprocessing weather data...")
    weather_daily = preprocess_weather_data(weather)

    print("Preprocessing consumption data...")
    consumption_daily = preprocess_consumption_data(consumption, clustering_results)

    print("Preparing dataset for classification...")
    all_data_daily = prepare_classification_dataset(
        consumption_daily, weather_daily, socioeconomic, cadaster
    )

    all_data_daily = all_data_daily.dropna(subset=["cluster"])


    print("Training Decision Tree Classifier...")
    X = all_data_daily.drop(["date", "postalcode", "cluster", "consumption"], axis=1)
    y = all_data_daily["cluster"]
    clf, feature_names = train_decision_tree_classifier(X, y)

    print("Exporting Decision Tree...")
    visualize_decision_tree(clf, feature_names)

