
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np

# Function to transform the time data (convert to local time, extract hour and date)
def transform_time(consumption):
    # Convert 'time' column to datetime format if not already converted
    consumption['time'] = pd.to_datetime(consumption['time'])

    # Check if 'time' already has a timezone. Use tz_convert if it does; otherwise, apply tz_localize
    if consumption['time'].dt.tz is None:
        consumption['localtime'] = consumption['time'].dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid')
    else:
        consumption['localtime'] = consumption['time'].dt.tz_convert('Europe/Madrid')

    # Extract hour and date from the 'localtime' column
    consumption['hour'] = consumption['localtime'].dt.hour
    consumption['date'] = consumption['localtime'].dt.date

    # Sort by postal code and local time
    consumption = consumption.sort_values(by=['postalcode', 'localtime'])

    return consumption

# Function to calculate rolling statistics (mean, std, quantile)
def calculate_rolling_stats(consumption):
    # Rolling mean over 48 hours
    consumption['rolling_mean'] = consumption.groupby('postalcode')['consumption'].rolling(window=48, min_periods=4, center=True).mean().reset_index(level=0, drop=True)
    # Rolling std over 48 hours
    consumption['rolling_std'] = consumption.groupby('postalcode')['consumption'].rolling(window=48, min_periods=4, center=True).std().reset_index(level=0, drop=True)
    # Rolling 10th quantile over 168 hours
    consumption['rolling_q10'] = consumption.groupby('postalcode')['consumption'].rolling(window=168, min_periods=24, center=True).quantile(0.1).reset_index(level=0, drop=True)
    return consumption

# Function to normalize the consumption data (z-score)
def normalize_z(consumption):
    # Z-normalization of the consumption data
    consumption['z_norm'] = (consumption['consumption'] - consumption['rolling_mean']) / consumption['rolling_std']
    return consumption

# Function to filter extreme values based on z-score
def filter_extreme_values(consumption, max_threshold=4, min_threshold=-2):
    # Filter out extreme consumption values based on z_norm and rolling quantile
    consumption['consumption_filtered'] = np.where(
        (consumption['z_norm'] < max_threshold) & 
        (consumption['z_norm'] > min_threshold) &
        (consumption['consumption'] > consumption['rolling_q10'] * 0.7),
        consumption['consumption'], 
        np.nan
    )
    return consumption

# Function to transform and group the cadaster data by postalcode
def transform_cadaster_data(cadaster):
    
    cadaster['centroid'] = cadaster.geometry.centroid
    # Rename columns for clarity
    cadaster = cadaster.rename(columns={
        "numberOfDwellings": "households",
        "reference": "cadastralref",
        "value": "builtarea",
        "currentUse": "currentuse",
        "conditionOfConstruction": "condition"
    })

    cadaster = cadaster[["postalcode", "currentuse", "condition", "households", "cadastralref", "builtarea"]]

    # Filter out non-residential buildings and non-functional conditions
    cadaster = cadaster[cadaster['currentuse'] == "1_residential"]
    cadaster = cadaster[cadaster['condition'] == "functional"]
    cadaster = cadaster.dropna(subset=['postalcode'])
    
    # Group by postal code and sum builtarea and households
    cadaster_grouped = cadaster.groupby('postalcode')[['builtarea', 'households']].sum().reset_index()
    return cadaster_grouped

# Function to join cadaster data with postal code geometries
def join_cadaster_with_postalcode(cadaster, postalcodes):
    postalcodes = postalcodes.rename(columns={"PROV": "provincecode", "CODPOS": "postalcode"})
    # Ensure both GeoDataFrames have the same CRS
    postalcodes = postalcodes.to_crs(cadaster.crs)
    # Perform spatial join to match cadaster centroids with postal code polygons
    cadaster = gpd.sjoin(
        cadaster, 
        postalcodes[['postalcode', 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    return cadaster

# Main function to apply all transformations to the consumption and cadaster data
def transform_consumption_data(consumption, cadaster, postalcodes):
    # Apply all transformations to the consumption data
    consumption = transform_time(consumption)
    consumption = calculate_rolling_stats(consumption)
    consumption = normalize_z(consumption)
    consumption = filter_extreme_values(consumption)

    # Transform cadaster data and join with postal codes
    cadaster_with_postalcode = join_cadaster_with_postalcode(cadaster, postalcodes)
    cadaster_grouped = transform_cadaster_data(cadaster_with_postalcode)

    # Return the transformed consumption and cadaster data
    return consumption, cadaster_grouped

