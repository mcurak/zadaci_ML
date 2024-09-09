import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

image_file = 'C:/Users/matea/OneDrive/Desktop/listLabs-zadatak/response_bands.TIFF'
with rasterio.open(image_file) as src:
    band_red = src.read(4)  # kanal 4 RED band
    band_nir = src.read(8)  #  kanal 8 NIR band
    band_swir = src.read(11)  #  kanal 8 SWIR band 
    
#red_band = rasterio.open('C:/Users/matea/OneDrive/Desktop/listLabs-zadatak/response_bands.TIFF').read(4)
#nir = rasterio.open('C:/Users/matea/OneDrive/Desktop/listLabs-zadatak/response_bands.TIFF').read(8)
#swir = rasterio.open('C:/Users/matea/OneDrive/Desktop/listLabs-zadatak/response_bands.TIFF').read(11)

# Calculate NDVI
np.seterr(divide='ignore', invalid='ignore')
ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)
# Calculate NDMI
ndmi = (band_nir.astype(float) - band_swir.astype(float)) / (band_nir + band_swir)



# Save the NDVI image
ndvi_image = 'ndvi_image.tif'
with rasterio.open(
    ndvi_image,
    'w',
    driver='GTiff',
    height=ndvi.shape[0],
    width=ndvi.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=src.crs,
    transform=src.transform,
) as dst:
    dst.write(ndvi, 1)
    
# Save the NDMI image
ndmi_image = 'ndmi_image.tif'
with rasterio.open(
    ndmi_image,
    'w',
    driver='GTiff',
    height=ndmi.shape[0],
    width=ndmi.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=src.crs,
    transform=src.transform,
) as dst:
    dst.write(ndmi, 1)


num_bands = src.count
print(f'Satelitska snimka sadrži: {num_bands} kanala')

average_ndvi = round(np.nanmean(ndvi),2)
print(f'Average NDVI: {average_ndvi}')
average_ndmi = round(np.nanmean(ndmi),2)
print(f'Average NDMI: {average_ndmi}')




plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title('NDVI')
plt.show()

plt.imshow(ndmi, cmap='RdYlGn')
plt.colorbar()
plt.title('NDMI')
plt.show()
