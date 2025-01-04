import os
import sys
import pandas as pd
import geopandas as gpd

from data_transformation.main import transform_consumption_data
from identify_load_curves_1.main import identify_load_curves

#ENVIROMENT VALUES
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

#DATA PATH
WD_DATA = os.path.join(BASE_PATH,"data")
ELECTRICITITY_DATA     = os.path.join(WD_DATA,"electricity_consumption.parquet")
WEATHER_DATA           = os.path.join(WD_DATA,"weather.parquet")
SOCIOECONOMIC          = os.path.join(WD_DATA,"socioeconomic.parquet")
CADASTER_LLEIDA_DATA   = os.path.join(WD_DATA,"cadaster_lleida.gml")
CADASTER_ALCARRAS_DATA = os.path.join(WD_DATA,"cadaster_alcarras.gml")
CADASTER_ALPICAT_DATA  = os.path.join(WD_DATA,"cadaster_alpicat.gml")
POSTAL_CODES_LLEIDA    = os.path.join(WD_DATA,"postal_codes_lleida.gpkg")

# Read the datasets with pandas
CONSUMPTION = pd.read_parquet(ELECTRICITITY_DATA)
WEATHER = pd.read_parquet(WEATHER_DATA)
SOCIOECONOMIC = pd.read_parquet(SOCIOECONOMIC)

# Concatenate GML files with geopandas
CADASTER = pd.concat([
    gpd.read_file(CADASTER_LLEIDA_DATA),
    gpd.read_file(CADASTER_ALCARRAS_DATA),
    gpd.read_file(CADASTER_ALPICAT_DATA)
])

# Concatenate GML files with geopandas
POSTALCODES = gpd.read_file(POSTAL_CODES_LLEIDA)

#DATA TRANSFORM
CONSUMPTION, CADASTER_WITH_POSTALCODE = transform_consumption_data(CONSUMPTION, CADASTER, POSTALCODES)

if __name__ == "__main__":
    identify_load_curves(
        CONSUMPTION,
        scaling_method="z_norm_scaling", 
        n_clusters=4, 
        do_silhouette = False
    )