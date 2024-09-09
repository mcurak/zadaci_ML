import requests
from collections import Counter
import geopandas as gpd
import pandas as pd
import geojson


feature_collection = gpd.read_file("https://plovput.li-st.net/getObjekti/")
# Count the number of features
feature_count = len(feature_collection)
print(f'Number of objects: {feature_count}')



url = "https://plovput.li-st.net/getObjekti/"
response = requests.get(url)
data = response.json()

filtered_features = [feature for feature in data['features'] if feature['properties']['tip_objekta'] ==16]
feature_counter = len(filtered_features)
print(f'Number of objects with type 16: {feature_counter}')


new_geojson = geojson.FeatureCollection(filtered_features)

# Step 4: Save the new GeoJSON to a file
with open('filtered_data.geojson', 'w') as f:
    geojson.dump(new_geojson, f)

#print(new_geojson)

