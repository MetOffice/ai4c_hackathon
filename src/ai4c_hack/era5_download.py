import cdsapi
import pathlib

dataset = "reanalysis-era5-pressure-levels"

var_list = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ]

request = {
    "product_type": ["reanalysis"],
    "variable": var_list,
    "year": ["2010"],
    "month": ["01"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "pressure_level": [
        "200", "500", "750",
        "800", "1000"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [-23, 17, -35, 35]
}

client = cdsapi.Client()

data_dir = pathlib.Path('/gws/ssde/j25a/mmh_storage/ai4c_data/era5_pl/')
current_month = 1
for var_name in var_list:
    fname = f'era5_{var_name}_{current_month:02d}.nc'
    request['variable'] = [var_name]
    request['month'] = [f'{current_month:02d}']
    req1 = client.retrieve(dataset,
                    request,
                    str(data_dir / fname),
                   )
