import os
import pandas as pd
import geopandas as gpd

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

'''
# Renombrar columnas del dataframe de códigos postales
postalcodes.rename({"PROV": "provincecode", "CODPOS": "postalcode"}, axis=1, inplace=True)

# Calcular el centroid del cadaster
cadaster["centroid"] = cadaster.geometry.centroid

# Convertir postalcodes al sistema de coordenadas de cadaster
postalcodes = postalcodes.to_crs(cadaster.crs)

# Realizar un join espacial entre cadaster y postalcodes
cadaster = gpd.sjoin(
    cadaster,
    postalcodes[["postalcode", "geometry"]],
    how="left",
    predicate="within"
)

# Seleccionar y renombrar columnas relevantes
cadaster = cadaster[
    ["postalcode", "currentUse", "conditionOfConstruction", "numberOfDwellings", "reference", "value"]
]
cadaster.rename({
    "numberOfDwellings": "households",
    "reference": "cadastralref",
    "value": "builtarea",
    "currentUse": "currentuse",
    "conditionOfConstruction": "condition"
}, axis=1, inplace=True)

# Filtrar el dataframe
cadaster = cadaster[
    ~pd.isna(cadaster.postalcode) &
    (cadaster.currentuse == "1_residential") &
    (cadaster.condition == "functional")
]

# Agrupar y sumar builtarea y households por código postal
cadaster_grouped = cadaster.groupby("postalcode")[["builtarea", "households"]].sum().reset_index()

# Convertir el dataframe agrupado a un DataFrame estándar de pandas
cadaster_final = pd.DataFrame(cadaster_grouped)

# Resultado final
print(cadaster_final)
'''

if __name__ == "__main__":
    identify_load_curves(CONSUMPTION)