import rasterio
from rasterio.warp import reproject, Resampling
import os
import numpy as np
import xarray
import rioxarray
import matplotlib.pyplot as plt
import regionmask
import pandas as pd

def Spatial_Aggregation(res,roi,data_input,data_resampled):
    
    ### OPEN INPUT TIF FILES RIOXARRAY (xarray extension, powered by rasterio)TO SELECT THE ATTRIBUTES FROM THE METADATA 
    
    data = rioxarray.open_rasterio(data_input, masked=True)

    ### read their metadata in the band_data inside the Data variables to select the physical variables: 
    
    conversion=data.eStation2_conversion
    scaling_factor=data.eStation2_scaling_factor
    nodata=data.eStation2_nodata
    scaling_offset=data.eStation2_scaling_offset

    

    fv = nodata

    dst_crs = {'init': 'EPSG:4326'}


    #### Define the ROI: Coordinates
    
    south = roi[0]
    north = roi[1]
    west = roi[2]
    east = roi[3]

    dst_lat = np.arange(south, north, res) + res / 2
    dst_lon = np.arange(west, east, res) + res / 2

    dst_shape = (dst_lat.size, dst_lon.size)
    
    ### resample data to target shape using resolution factor and scale data transform


    dst_transform = rasterio.transform.from_bounds(
        west=dst_lon.min() - res / 2,
        south=dst_lat.min() - res / 2,
        east=dst_lon.max() + res / 2,
        north=dst_lat.max() + res / 2,
        width=dst_shape[1],
        height=dst_shape[0]
    )

    #### Rasterio can map the pixels of a destination raster with an associated coordinate reference system 
    #### and transform to the pixels of a source image with a different coordinate reference system and transform. 
    #### This process is  known as reprojection   
    #### Resampling method: average 
    
    
    with rasterio.open(data_input) as src:
        data, transform = reproject(source=src.read(),
                                    destination=np.zeros(dst_shape),
                                    src_transform=src.transform,
                                    dst_transform=dst_transform,
                                    src_crs=src.crs,
                                    dst_crs=src.crs,
                                    dst_nodata=fv,
                                    src_nodata=fv,
                                    resampling=Resampling.average)
        data = data.astype('float')
        data[data == fv] = np.nan
        data *= scaling_factor
    
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': dst_shape[1],
            'height': dst_shape[0]
        })
 
        with rasterio.open(
            data_resampled,
            'w',
            driver='GTiff',
            height=dst_shape[0],
            width=dst_shape[1],
            count=1,
            dtype=np.float32,
            crs=dst_crs,
            transform=dst_transform,
        ) as dest_file:
            dest_file.write(data, 1)
            dest_file.close()

            
    resampled = xarray.open_dataset(data_resampled)
    print('Resampled shape:',resampled.rio.shape)
    print('Resampled resolution:',resampled.rio.resolution())
    print('Resampled resampled bounds:',resampled.rio.bounds())

    
    return resampled


def Mask_Area_Plot(resampled,file_roi,resampled_mask):

    t1 = xarray.open_dataset(resampled)
   
    k_rename1=t1.band_data.rename({'x': 'lon','y': 'lat'})
    
    ts1 = k_rename1[:,:,:]

    ts1[:]
    ts1_mask= np.mean(ts1[:],axis = 0) 
    
    lon_name_ts1   = ts1.lon[:]
    lat_name_ts1   = ts1.lat[:]

    
    outline_africa = np.array(file_roi)

    region_area_africa = regionmask.Regions([outline_africa])
    
    mask_pygeos_area_ts1 = region_area_africa.mask(k_rename1, method="pygeos") 
    
    LON, LAT = np.meshgrid(lon_name_ts1, lat_name_ts1)
    
    ts1_area = ts1_mask.values
    ts1_area[np.isnan(mask_pygeos_area_ts1)] = np.nan 
    ts1_mask.rio.set_crs("epsg:4326")
    ts1_mask.rio.set_spatial_dims("lon", "lat", inplace=True)
    
    ds = ts1_mask.rio.write_crs("epsg:4326")
    
    ds.rio.to_raster(resampled_mask)
    

    #plt.savefig('save.png')

    return ts1_area