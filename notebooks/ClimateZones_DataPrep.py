#!/usr/bin/env python
# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
import pathlib
import os
import json

import matplotlib.pyplot
import cartopy.crs

import xarray
import pandas

import multiprocessing

with open ('config.json','r') as tutorial_config:
    tutorial_config = json.load(tutorial_config)
tutorial_config


def get_platform_dir(select_platform):
    if select_platform == 'mo_linux':
      return pathlib.Path(os.environ['SCRATCH']) / 'climate_zones'
    if select_platform == 'jasmin':
        return pathlib.Path('/gws/nopw/j04/mohc_shared/dscop/') / 'climate_zones'
    print('platform not found, return generic path')
    return pathlib.Path(os.environ['HOME']) / 'climate_zones',


def get_data_path(root_dir, time_period, scenario_id, prefix, resolution_str, suffix) :
    start_year = time_period[0]
    end_year = time_period[1]
    if scenario_id == historic_scenario_str:
        data_dir = root_data_dir / time_dir_template.format(start_year=start_year,end_year=end_year)
    else:
        data_dir = root_data_dir / time_dir_template.format(start_year=start_year,end_year=end_year) / scenario_id 
    data_fname = fname_template.format(prefix=prefix, 
                                       res=resolution_str, 
                                       suffix=format_str)
    return data_dir / data_fname

def create_dataframe_period(data_period_scenario, period, scenario):
    print(f'processing period {period} scenario {scenario}')
    df_mean = data_period_scenario['climate_mean'].to_dataframe().unstack(['time'])
    df_mean.columns = ['_'.join(map(str,l1))+ '_mean' for l1 in df_mean.columns]
    df_mean = df_mean.reset_index()

    df_std = data_period_scenario['climate_std'].to_dataframe().unstack(['time'])
    df_std.columns = ['_'.join(map(str,l1)) + '_std' for l1 in df_std.columns]
    df_std = df_std.reset_index()

    df_zone = data_period_scenario['climate_zone'].to_dataframe()
    df_zone = df_zone.reset_index()

    df_climate_zones = df_mean.merge(df_std, on=['lat','lon'])
    df_climate_zones = df_climate_zones.merge(df_zone, on=['lat','lon'])

    df_climate_zones = df_climate_zones[~df_climate_zones['precipitation_1.0_mean'].isna()]
    df_climate_zones['period_start'] = period[0]
    df_climate_zones['period_end'] = period[1] 
    df_climate_zones['scenario'] = scenario

    df_climate_zones['climate_group'] = df_climate_zones['kg_class'].apply(lambda v1: climate_group_lookup[int(v1)])
    df_climate_zones['climate_subgroup'] = df_climate_zones['kg_class'].apply(lambda v1: climate_subgroup_lookup[int(v1)])

    return df_climate_zones

def process_period_scenario(path_dict, current_res, current_period, current_scenario, out_path):
    print(f'processing segment {current_period} - {current_scenario}')
    current_data = {ds_id: xarray.open_dataset(current_path[current_res]) for ds_id, current_path in path_dict.items()}
    # process into a dataframe
    current_df = create_dataframe_period(
        current_data, 
        current_period,
        current_scenario)
    current_df.to_csv(out_path, index=False)
    return current_df

current_platform = tutorial_config['platform']

# root_data_dir = get_platform_dir(current_platform)
root_data_dir = pathlib.Path(os.environ['SCRATCH']) / 'climate_zones'
ml_ready_output_dir = root_data_dir / 'arco'
print(f'output dir {ml_ready_output_dir}')
    
resolutions_dict = {float(k1): v1 for k1,v1 in tutorial_config['resolutions_names'].items()}
dataset_prefix_dict = tutorial_config['dataset_prefix']

format_str = 'nc'
historic_scenario_str = 'historic'

future_scenario_list = tutorial_config['future_scenarios']
historic_scenario_list = tutorial_config['historic_scenarios']

time_periods = { 
    (1901,1930): historic_scenario_list, 
    (1931,1960): historic_scenario_list,
    (1961,1990): historic_scenario_list,
    (1991,2020): historic_scenario_list,
    (2041,2070): future_scenario_list,
    (2071,2099): future_scenario_list,
}

fname_template = tutorial_config['fname_template']
time_dir_template = tutorial_config['time_dir_template']
csv_out_template = tutorial_config['csv_out_template']
period_scenario_template = 'climate_zones_{start}_{end}_{scenario}_{res}.csv'


climate_subgroups_dict = tutorial_config['climate_subgroups']#
climate_subgroup_lookup = [k1 for k1 in climate_subgroups_dict.keys()]

climate_group_lookup = [k1[0] for k1 in climate_subgroups_dict.keys()]
climate_group_lookup[0] = 'none'


data_path_dict = {
    (start_year, end_year): {
        scenario_id: { ds_id: { current_res: get_data_path(root_dir=root_data_dir,
                                                           time_period=(start_year, end_year),
                                                           scenario_id=scenario_id, 
                                                           prefix=ds_str,
                                                           resolution_str=res_str,
                                                           suffix=format_str,
                                                          )
                                for current_res, res_str in resolutions_dict.items() 
                              }
                       for ds_id, ds_str in dataset_prefix_dict.items()
                     } 
        for scenario_id in current_scenarios
    }
    for (start_year, end_year), current_scenarios in time_periods.items()                                                                                                             
}

inter_paths = {}
pool_args_list = {}
for current_res, res_str in resolutions_dict.items():
    climate_zones_df_list = []
    inter_paths[current_res]= []
    res_args = []
    for current_period, scenario_paths in data_path_dict.items():
        for current_scenario, scenario_data in scenario_paths.items():
            out_path = ml_ready_output_dir / period_scenario_template.format(res=resolutions_dict[current_res],
                                                                             start=current_period[0],
                                                                             end=current_period[1],
                                                                             scenario=current_scenario,
                                                                            )
            inter_paths[current_res] += [out_path]
            res_args  += [(scenario_data,
                                current_res,
                                current_period,
                                current_scenario,
                                out_path,
                               )]
    pool_args_list[current_res] = res_args

use_multiprocessing = False
print('starting data processing')

if use_multiprocessing:
    spice_pool = multiprocessing.Pool(4)
    spice_pool


for current_res, res_str in reversed(resolutions_dict.items()):
    print(current_res)
    if use_multiprocessing:
        res_it = spice_pool.starmap(process_period_scenario, pool_args_list[current_res])
        #trigger execution, but throw away data and instead read from disk, to reduce memory usage.
        period_df_list = [df_ps for df_ps in res_it]
        climate_zones_merged_df = pandas.concat([pandas.read_csv(path1) for path1 in inter_paths[current_res]]).reset_index().drop(['index'],axis='columns')
    else:
        print('serial processing')
        scenario_df_list = []
        for current_args in pool_args_list[current_res]:
            scenario_df_list += [process_period_scenario(*current_args)]
        climate_zones_merged_df = pandas.concat(scenario_df_list).reset_index().drop(['index'],axis='columns')
    # save out merged dataframe to disk
    out_path = ml_ready_output_dir / csv_out_template.format(resolution=resolutions_dict[current_res])
    print(out_path)
    climate_zones_merged_df.to_csv(out_path, index=False)
