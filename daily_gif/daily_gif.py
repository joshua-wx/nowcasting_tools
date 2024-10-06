import os
import re
import gzip
import time
import argparse
import tempfile
from glob import glob
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import cartopy
from cartopy import crs as ccrs # A toolkit for map projections
import pickle

import dask.bag as db

import pyart
matplotlib.use('agg')

def create_image(odim_ffn, image_folder, radar_type, sweep_idx,
                    dbz_name, zdr_name, phidp_name, width_name, rhohv_name):

    """
    Plots all ocean pol volumes for a given date
    """
    
    ref_min = 0
    ref_max = 60
    figsize = (8,8)
    #read date from filename and volume
    if radar_type=='cp2':
        dtstr = re.findall('[0-9]{8}_[0-9]{6}', os.path.basename(odim_ffn))
        odim_dt = datetime.strptime(dtstr[0], '%Y%m%d_%H%M%S')
        radar = pyart.io.read_mdv(odim_ffn, file_field_names=True)
    elif radar_type=='opol':
        dtstr = re.findall('[0-9]{8}-[0-9]{6}', os.path.basename(odim_ffn))
        odim_dt = datetime.strptime(dtstr[0], '%Y%m%d-%H%M%S')
        radar = pyart.aux_io.read_odim_h5(odim_ffn, file_field_names=True)
    lat, lon, _ = radar.get_gate_lat_lon_alt(sweep_idx)

    #setup figure
    fig = plt.figure(figsize=figsize, facecolor='w')
    ax = fig.add_subplot(111, projection= ccrs.PlateCarree())
    ax.set_extent([np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)],crs=ccrs.PlateCarree())

    #add city markers
    fname = cartopy.io.shapereader.natural_earth(resolution='10m', category='cultural', name='populated_places')
    reader = cartopy.io.shapereader.Reader(fname)
    city_list = list(reader.records())
    for city in city_list:
        if (((city.attributes['LATITUDE'] >= np.amin(lat)) and (city.attributes['LATITUDE'] <= np.amax(lat)))
            and ((city.attributes['LONGITUDE'] >= np.amin(lon)) and (city.attributes['LONGITUDE'] <= np.amax(lon)))):
            ax.scatter(city.attributes['LONGITUDE'], city.attributes['LATITUDE'], s=6, color='black',
                       transform=ccrs.PlateCarree(), zorder=5)
            ax.text(city.attributes['LONGITUDE']+0.01, city.attributes['LATITUDE']+0.01, 
                    city.attributes['NAME'], fontsize=10,transform=ccrs.PlateCarree())
    #plot radar data        
    display = pyart.graph.RadarMapDisplay(radar)
    title_str = radar_type + ' Reflectivity ' + odim_dt.strftime('%H:%M %Y%m%d') + ' UTC'
    display.plot_ppi_map(dbz_name, sweep_idx, vmin=ref_min, vmax=ref_max,
                        cmap=pyart.graph.cm_colorblind.HomeyerRainbow, colorbar_flag = False,
                        resolution='10m', title_flag=False)

    ax.set_title(label=title_str, fontsize=16)

    #annotations
    display.plot_range_rings([50,100], ax=None, col='k', ls='-', lw=0.5)
    display.plot_point(radar.longitude['data'], radar.latitude['data'])
    #Now we add lat lon lines
    gl = display.ax.gridlines(draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5,
                              linestyle='--')
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xlabels_top = False
    gl.ylabels_right = False

    #here is our pretty colorbar code
    cbax = fig.add_axes([0.15, 0.06, 0.725, .025])
    cb = plt.colorbar(display.plots[0], cax=cbax, orientation='horizontal')
    cb.set_label(label='Reflectivity (dBZ)', fontsize=14)
    cb.ax.tick_params(labelsize=10)

    #save to file
    ffn_datestr = odim_dt.strftime('%Y%m%d_%H%M%S')
    png_ffn = f'{image_folder}/{radar_type}_{ffn_datestr}.jpeg'
    plt.savefig(png_ffn, optimize=True, quality=80)
    fig.clf()
    plt.close()
    if VERBOSE:
        print('SUCCESS, rendered:', png_ffn)
    
    return odim_dt

def make_gif(files, output, delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    INPUT:
        files: list of full file paths
        output: full filename for output gif
        delay: delay in ms between animation frames
        repeat: Set to infinite loop
    OUTPUT:
        None
    """
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))        
    
def worker(vol_path, gif_folder, radar_type, sweep_idx,
            dbz_name, zdr_name, phidp_name, width_name, rhohv_name):
    
    #create temp folder
    image_folder = tempfile.mkdtemp()
    
    #list vols
    if radar_type=='cp2':
        vol_file_list = sorted(glob(vol_path + '/*.mdv'))
    elif radar_type=='opol':
        vol_file_list = sorted(glob(vol_path + '/*.hdf'))
    else:
        print('radar_type not recognised')
    
    if len(vol_file_list) == 0:
        print('no vols in', vol_path)
        return None
    
    #generate images
    if VERBOSE:
        print('processing:', vol_path)
    for vol_file in vol_file_list:
        try:
            create_image(vol_file, image_folder, radar_type, sweep_idx,
                        dbz_name, zdr_name, phidp_name, width_name, rhohv_name)
        except Exception as e:
            print('IMG Failed on', vol_file, 'with', e)

    #convert to gif
    try:
        gif_datestr = vol_path[-8:]
        gif_ffn = f'{gif_folder}/{radar_type}_{gif_datestr}_reflectivity.gif'
        img_file_list = sorted(glob(image_folder + '/*.jpeg'))
        make_gif(img_file_list, gif_ffn, delay=25, repeat=True)
    except Exception as e:
        print('GIF Failed on', gif_ffn, 'with', e)
    
    #remove temp image folder
    os.system('rm -rf ' + image_folder)
    
    return None

def main(vol_root, gif_folder, radar_type, sweep_idx,
            dbz_name, zdr_name, phidp_name, width_name, rhohv_name):

    #build folder list
    folder_list = sorted(glob(vol_root + '/*'))
    
    #build args
    argslist = []
    for i, vol_folder in enumerate(folder_list):
        argslist.append([vol_folder, gif_folder, radar_type, sweep_idx, dbz_name, zdr_name, phidp_name, width_name, rhohv_name])

    #run matching
    bag = db.from_sequence(argslist).starmap(worker)
    bag.compute()
    
    return None

if __name__ == '__main__':
    
    VERBOSE = True
    
    # Parse arguments
    parser_description = "Unpack tar, convert to odim for daily rapic tar files"
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-j',
        '--cpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of process')
    parser.add_argument(
        '-i',
        '--input',
        dest='input_folder',
        default=None,
        type=str,
        help='vol root folder',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        dest='output_folder',
        default=None,
        type=str,
        help='gif output folder',
        required=True)
    parser.add_argument(
        '-t',
        '--radar_type',
        dest='radar_type',
        default=None,
        type=str,
        help='radar type',
        required=True)
    
    args = parser.parse_args()
    vol_root     = args.input_folder
    gif_folder   = args.output_folder
    radar_type   = args.radar_type
    
    #default settings
    if radar_type == 'cp2':
        sweep_idx=1
        dbz_name='DBZ'
        zdr_name='ZDR'
        phidp_name='PHIDP'
        width_name='WIDTH'
        rhohv_name='RHOHV'
        main(vol_root, gif_folder, radar_type, sweep_idx, dbz_name, zdr_name, phidp_name, width_name, rhohv_name)
    elif radar_type == 'opol':
        sweep_idx=2
        dbz_name='DBZH'
        zdr_name='ZDR'
        phidp_name='PHIDP'
        width_name='WRADH'
        rhohv_name='RHOHV'
        main(vol_root, gif_folder, radar_type, sweep_idx, dbz_name, zdr_name, phidp_name, width_name, rhohv_name)
    else:
        print('radar type unknown')
        
    print('finished')
    

#%run daily_gif.py -i /g/data/hj10/admin/cp2/level_1/s_band/sur/2014 -o /g/data/hj10/admin/cp2/level_1/s_band/daily_gif -t cp2