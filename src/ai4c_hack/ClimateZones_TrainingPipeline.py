# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.import pathlib
import os
import datetime
import json

import pandas

import sklearn
import sklearn.preprocessing
import sklearn.tree
import torch


def get_platform_dir(select_platform, config):
    try:
        root_path = pathlib.Path(config['default_dirs'][select_platform]) / 'climate_zones'
    except KeyError:
        root_path = pathlib.Path(os.environ['HOME']) / 'climate_zones'
    return root_path


print('start of training script')

with open ('config.json','r') as tutorial_config:
    tutorial_config = json.load(tutorial_config)
tutorial_config
current_platform = tutorial_config['platform']
print(f'config loaded, platform is {current_platform}')

root_data_dir = get_platform_dir(current_platform, tutorial_config)
print(f'data dir {root_data_dir}')

ml_ready_dir = root_data_dir / 'ml_ready'

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
ml_ready_fname_template = tutorial_config['csv_out_template']

current_res = 1.0

mlready_data_path = ml_ready_dir / ml_ready_fname_template.format(resolution=resolutions_dict[current_res])
print('loading data')
zones_df = pandas.read_csv(mlready_data_path)

# reducing the total data point to decrease memeory requirements
# zones_df = zones_df[zones_df['scenario'] == 'historic']
zones_df = zones_df[(zones_df['period_start']==1991)&(zones_df['scenario']=='historic')]

print('data loaded successfuly: ', zones_df.shape)

predictors_dict = {
    'precip_mean': [c1 for c1 in zones_df.columns if 'precipitation' in c1 and 'mean' in c1],
    'precip_std': [c1 for c1 in zones_df.columns if 'precipitation' in c1 and 'std' in c1],
    'temp_mean': [c1 for c1 in zones_df.columns if 'air_temperature' in c1 and 'mean' in c1],
    'temp_std': [c1 for c1 in zones_df.columns if 'air_temperature' in c1 and 'std' in c1],
}

predictors = predictors_dict['precip_mean'] + predictors_dict['temp_mean']

target_var = 'climate_group' # 5 classes
# target_var = 'climate_subgroup # 30 classes


# Train/val/test split

random_seed = tutorial_config['random_seed']

test_frac = 0.1
val_frac = 0.1
val_frac_sub = (val_frac / (1.0-test_frac) )

test_df = zones_df.groupby(['period_start','scenario']).sample(frac=test_frac, random_state=random_seed)
remain_df = zones_df.drop(test_df.index)

val_df = remain_df.groupby(['period_start','scenario']).sample(frac=val_frac_sub, random_state=random_seed)
train_df = remain_df.drop(val_df.index)

print('train: ',train_df.shape)
print('validation: ',val_df.shape)
print('test: ',test_df.shape)

# Normalising the data

input_scaler = sklearn.preprocessing.StandardScaler()
input_scaler.fit(train_df[predictors])

X_train = input_scaler.transform(train_df[predictors])
X_val = input_scaler.transform(val_df[predictors])
X_test = input_scaler.transform(test_df[predictors])

target_encoder = sklearn.preprocessing.OneHotEncoder()
target_encoder.fit(train_df[[target_var]])

y_train = target_encoder.transform(train_df[[target_var]]).toarray()
y_val = target_encoder.transform(val_df[[target_var]]).toarray()
y_test = target_encoder.transform(test_df[[target_var]]).toarray()

experiment_name = 'ai4c_climate_zone'


classifiers_params = {
    'decision_tree': {'class': sklearn.tree.DecisionTreeClassifier, 'opts': {'max_depth':10, 'class_weight':'balanced'}},
    'random_forest': {'class': sklearn.ensemble.RandomForestClassifier, 'opts': {'max_depth':10, 'class_weight':'balanced', 'n_estimators': 10}},
     'ann_3_100': {'class': sklearn.neural_network.MLPClassifier, 'opts': {'hidden_layer_sizes':(100,100,100,)}},
     # 'ann_3_200': {'class': sklearn.neural_network.MLPClassifier, 'opts': {'hidden_layer_sizes':(200,200,200)}},   
}


print('starting training')

classifiers_dict = {}             
for clf_name, clf_params in classifiers_params.items():
    print(f'training algorithm {clf_name}')
    train_start = datetime.datetime.now()
    clf1 = clf_params['class'](**clf_params['opts'])
    clf1.fit(X_train, y_train)
    classifiers_dict[clf_name] = clf1
    print('training time: ', datetime.datetime.now() - train_start)
        
print('training complete')

y_pred_train = {}
y_pred_val = {}
print('starting inference')
for clf_name, clf_obj in classifiers_dict.items():
    y_pred_train[clf_name] = clf_obj.predict(X_train)
    y_pred_val[clf_name]= clf_obj.predict(X_val)

for clf_name, clf_obj in classifiers_dict.items():
    sklearn.metrics.precision_recall_fscore_support(y_train, y_pred_train[clf_name])


for clf_name, clf_obj in classifiers_dict.items():
    sklearn.metrics.precision_recall_fscore_support(y_val, y_pred_val[clf_name])

print('inference complete')
