#!/usr/bin/env python
# coding: utf-8



## import of libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from IPython.core.pylabtools import figsize

# additional tools needed
from re import match
from functools import reduce

## models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# sklearn additional libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.metrics import roc_auc_score

# for Hypterparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In this project, we're gonna classify if Radiant or Dire would be the winner using several features.
# This will focus on a Pre-game Classifier
# 
# # DOTA 2 RADIANT OR DIRE?
# To predict which team is gonna win before the game begins, we're gonna check and use the following features:
# * Day_of_Week
# * TimeHour
# * Region
# * Dire vs Radiant:
#     - Heroes Selected
#     - Winrate of players
#     - TrueSkill of players
# 

# ## Data Preparation
import os
from os.path import join

source_folder = './datasource'

# Import all the csv files and save the filenames to datasets
available_datasets = dict([(path[:-4], path) for path in os.listdir(source_folder) if path[-4:] == '.csv'])
available_datasets = dict(sorted(available_datasets.items()))
available_datasets

# Import the only needed csv files
cluster_regions_dataset = pd.read_csv(source_folder + '/' + 'cluster_regions.csv')
hero_names_dataset = pd.read_csv(source_folder + '/' + 'hero_names.csv')
match_dataset = pd.read_csv(source_folder + '/' + 'match.csv')
player_ratings_dataset = pd.read_csv(source_folder + '/' + 'player_ratings.csv')
players_dataset = pd.read_csv(source_folder + '/' + 'players.csv')

# To achieve the desired dataset, 4 sub-datasets will be prepared
# * match_dataset
# * player_ratings_dataset
# * players_dataset
# * hero_name

# ### Data Sources

# **Match Details**
match_dataset.head()

# Select the needed columns only
match_details = match_dataset[['match_id','start_time','game_mode','cluster','radiant_win']]
match_details

# Check if match_ids are unique in the dataset
match_details.match_id.is_unique

# Check the unique values of target variable 'radiant_win'
match_details['radiant_win'].unique()

# True - Radiant Win \
# False - Dire Win


# Check if there are null values in match_details
match_details.isna().sum()

match_details = pd.merge(match_details, cluster_regions_dataset, on ='cluster')
match_details = match_details.reset_index(drop=True)

del match_details['cluster']
match_details


# Checking on the start time, we are interested in the:
# * day of week
# * time of day
match_details['date'] = pd.to_datetime(match_details['start_time'],unit='s')
match_details['day_of_week'] = match_details['date'].dt.day_name()
match_details['time'] = match_details['date'].dt.strftime('%H:%M')
match_details['time'] = match_details['time'].str[:2].astype(int)

del match_details['start_time']
del match_details['date']

match_details


match_details['day_of_week'].value_counts(normalize=True)



hist_feature = 'time'
dataset = match_details[hist_feature]

# Create histogram
fig = plt.figure(tight_layout=True,
                 figsize=(7, 5)
                )

q25, q75 = np.percentile(dataset, [25, 75])
bin_width = 2 * (q75 - q25) * len(dataset) ** (-1/3)
bins = round((dataset.max() - dataset.min()) / bin_width)

plt.hist(dataset,
         color='blue',
         edgecolor='black',
         bins=bins)
 
plt.xlabel(hist_feature)
plt.ylabel('count')

plt.locator_params(axis='x', nbins=24)
fig.align_labels()

plt.title(hist_feature + ' histogram')

plt.show()


# Based from the histogram, we can see that the time of the day with highest count of players is between 2pm to 12 midnight

# Check the frequency of game_modes in the dataset
match_details['game_mode'].value_counts()

# 2 - Captain's Mode
# 22 - Ranked Matchmaking



match_details['game_mode'].value_counts(normalize=True)


# Since there's only 2.7% for captain mode games, we will remove it in the features we're gonna model with


del match_details['game_mode']


# Check the frequency of regions in the dataset
match_details['region'].value_counts(normalize=True)


match_details['region'].value_counts()


# To prevent very small amount of dataset for a given region, we will further regroup the regions


region_dict = {
        'EUROPE':'EMEA',
        'AUSTRIA':'EMEA',
        'STOCKHOLM':'EMEA',
        'DUBAI':'EMEA',
        'US EAST':'Americas',
        'US WEST':'Americas',
        'BRAZIL':'Americas',
        'CHILE': 'Americas',
        'PW TELECOM SHANGHAI': 'Asia',
        'PW TELECOM ZHEJIANG': 'Asia',
        'PW TELECOM GUANGDONG': 'Asia',
        'PW UNICOM': 'Asia',
        'SINGAPORE': 'Asia',
        'AUSTRALIA': 'Asia',
        'JAPAN': 'Asia'
    }



match_details['cluster'] = match_details['region'].map(region_dict)


print(match_details['cluster'].value_counts())
print(match_details['cluster'].value_counts(normalize=True))



match_details



match_ge_info = match_details[['match_id','region','cluster','day_of_week','time','radiant_win']]
match_ge_info = match_ge_info.sort_values(by='match_id').reset_index(drop=True)
match_ge_info


# **match_teamskills & match_winrates**


players_dataset


# Players dataset: Get the needed columns only
players = players_dataset[['match_id','account_id','hero_id','player_slot']]
players = players.reset_index(drop=True)
players


# This players dataset can be merge into player_ratings dataset which can be merge later on to match_details


player_ratings_dataset


# In this dataset, we are interested on the:
# * winrate (total_wins/total_matches)
# * trueskill_mu


# Players Ratings dataset: Get the needed columns only
player_ratings = player_ratings_dataset.copy()
player_ratings['winrate'] = player_ratings['total_wins'] / player_ratings['total_matches'] 

player_ratings = player_ratings[['account_id','winrate','trueskill_mu']]
player_ratings


hist_feature = 'winrate'
dataset = player_ratings[hist_feature]

# Create histogram
fig = plt.figure(tight_layout=True,
                 figsize=(7, 5)
                )

q25, q75 = np.percentile(dataset, [25, 75])
bin_width = 2 * (q75 - q25) * len(dataset) ** (-1/3)
bins = round((dataset.max() - dataset.min()) / bin_width)

plt.hist(dataset,
         color='blue',
         edgecolor='black',
         bins=bins)
 
plt.xlabel(hist_feature)
plt.ylabel('count')

plt.locator_params(axis='x', nbins=10)
fig.align_labels()

plt.title(hist_feature + ' histogram')

plt.show()



winrate_freq = player_ratings['winrate'].value_counts(bins = 5).sort_index()
winrate_freq



hist_feature = 'trueskill_mu'
dataset = player_ratings[hist_feature]

# Create histogram
fig = plt.figure(tight_layout=True,
                 figsize=(7, 5)
                )

q25, q75 = np.percentile(dataset, [25, 75])
bin_width = 2 * (q75 - q25) * len(dataset) ** (-1/3)
bins = round((dataset.max() - dataset.min()) / bin_width)

plt.hist(dataset,
         color='blue',
         edgecolor='black',
         bins=bins)
 
plt.xlabel(hist_feature)
plt.ylabel('count')

plt.locator_params(axis='x', nbins=10)
fig.align_labels()

plt.title(hist_feature + ' histogram')

plt.show()



trueskill_freq = player_ratings['trueskill_mu'].value_counts(bins = 5).sort_index()
trueskill_freq


# In order to have some categorization in the player ratings, we will use the ff categorization:
# * (4.949, 13.76]    Very Low Skill 
# * (13.76, 22.526]   Low Skill 
# * (22.526, 31.293]  Normal Skill 
# * (31.293, 40.059]  High Skill
# * (40.059, 48.826]  Very High Skill 


def get_skillrank(trueskill_mu):
        if trueskill_mu > 40.059:
             return "Very High Skill"
        elif trueskill_mu > 31.293:
             return "High Skill"
        elif trueskill_mu > 22.526:
             return "Normal Skill"
        elif trueskill_mu > 13.76:
             return "Low Skill"
        elif trueskill_mu > 0:
             return "Very Low Skill"
        else:
             return "Unknown"


player_ratings['skillrank'] = player_ratings['trueskill_mu'].apply(get_skillrank)
del player_ratings['trueskill_mu']


player_ratings


# Combine players and player_ratings dataset

player_stats = pd.merge(players,player_ratings, on = 'account_id')
player_stats = player_stats.sort_values(by=['match_id','player_slot']).reset_index(drop = True)

del player_stats['account_id']
del player_stats['hero_id']

print(player_stats[:20])
player_stats


# Include Team based on player_slot:
# * 0-4: Radiant
# * 128-132: Dire


player_stats_new = player_stats.copy()
player_stats_new['team'] = np.where(player_stats_new['player_slot'] <= 4, 'radiant', 'dire')
del player_stats_new['player_slot']

player_stats_new[:20]


player_stats_new[player_stats_new['match_id'] == 4]



player_stats_ref = player_stats_new.copy()
player_stats_ref['skillrank'] = player_stats_ref['team'] + '_' + player_stats_ref['skillrank'].str.lower()
match_teamskills = player_stats_ref.groupby(['match_id','team'])['skillrank'].aggregate(list).unstack().reset_index()
match_teamskills = match_teamskills[['match_id','radiant','dire']]
match_teamskills.columns = ['match_id','radiant_skills','dire_skills']
match_teamskills.head()



player_stats_ref = player_stats_new.copy()
match_winrates = player_stats_ref.groupby(['match_id','team'])['winrate'].mean().unstack().reset_index().rename_axis(None, axis=1)
match_winrates = match_winrates[['match_id','radiant','dire']]
match_winrates.columns = ['match_id','radiant_winrate','dire_winrate']
match_winrates.head()


# **Player Heroes**
# Heroes dataset: Get the needed columns only
heroes_df = hero_names_dataset[['hero_id','localized_name']]
heroes_df = heroes_df.rename(columns={'localized_name': 'hero_name'})
heroes_df = heroes_df.reset_index(drop=True)
hero_lookup = dict(zip(heroes_df['hero_id'], heroes_df['hero_name']))
heroes_df


print(heroes_df['hero_name'].values)


# Check if hero_ids in players dataset can be found in the heroes dataset

print('players_dataset')
players_hero_ids = players['hero_id'].unique().astype(int)
np.sort(players_hero_ids)


print('heroes_dataset')
hero_ids = heroes_df['hero_id'].values
np.sort(hero_ids)


# Since there is no 0 in the heroes dataset but present in players dataset, We will create 0 in the heroes_dataset and lookup

# heroes dataframe
new_heroes_df = heroes_df.copy()

list_row = [0,'Unknown']
new_heroes_df.loc[len(new_heroes_df)] = list_row

new_heroes_df = new_heroes_df.drop_duplicates()
new_heroes_df = new_heroes_df.sort_values(by='hero_id')
new_heroes_df = new_heroes_df.reset_index(drop = True)
new_heroes_df


# heroes lookup      
new_hero_lookup = list(hero_lookup.items())
new_hero_lookup.insert(0, (0, 'Unkown'))
new_hero_lookup = dict(new_hero_lookup)

new_hero_lookup



players



player_heroes = players.copy()
player_heroes['hero'] = player_heroes['hero_id'].map(hero_lookup)
player_heroes = player_heroes.sort_values(by=['match_id','player_slot']).reset_index(drop = True)

del player_heroes['account_id']
del player_heroes['hero_id']

player_heroes



match_team_heroes = player_heroes.copy()
match_team_heroes['team'] = np.where(match_team_heroes['player_slot'] <= 4, 'radiant', 'dire')
match_team_heroes['hero'] = match_team_heroes['team'] + '_' + match_team_heroes['hero'].str.lower()

del match_team_heroes['player_slot']

match_team_heroes[:20]



match_heroes = match_team_heroes.groupby(['match_id','team'])['hero'].aggregate(list).unstack().reset_index().rename_axis(None, axis=1)
match_heroes = match_heroes[['match_id','radiant','dire']]
match_heroes.columns = ['match_id','radiant_heroes','dire_heroes']
match_heroes.head()


# ### Merge all datasets
# 
# We will combine all the datasets we have preprocessed:
# * match_ge_info
# * match_teamskills
# * match_winrates
# * match_heroes

data_frames = [match_ge_info, match_winrates, match_teamskills, match_heroes]


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)



df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['match_id'],
                                            how='outer'), data_frames)
df_merged.head()



df_merged.isna().sum()


# Drop all records with NaN

df_merged_dropna = df_merged.dropna().reset_index(drop=True)
df_merged_dropna.head()


df_merged_dropna['len_radiant_skills']= df_merged_dropna['radiant_skills'].str.len()
df_merged_dropna['len_dire_skills']= df_merged_dropna['dire_skills'].str.len()
df_merged_dropna['len_radiant_heroes']= df_merged_dropna['radiant_heroes'].str.len()
df_merged_dropna['len_dire_heroes']= df_merged_dropna['dire_heroes'].str.len()
df_merged_dropna['min_list_cols'] = df_merged_dropna[['len_radiant_skills','len_dire_skills','len_radiant_heroes','len_dire_heroes']].min(axis=1)


df_merged_dropna.head()



df_merged_dropna.info()


incomplete_lists_count = len(df_merged_dropna[df_merged_dropna['min_list_cols'] < 5])
incomplete_lists_count


# If we drop all the records with less than 5 items in the lists for skills and heroes, the resulting dataset will significantly reduced to more than half. With that we will just do the following:
# * for skills bracket, if less than 5 increase the number of normal skills
# * for heroes, as-is

df_merged_new = df_merged_dropna

del df_merged_dropna['len_radiant_skills']
del df_merged_dropna['len_dire_skills']
del df_merged_dropna['len_radiant_heroes']
del df_merged_dropna['len_dire_heroes']
del df_merged_dropna['min_list_cols']

df_merged_new.info()



df_merged_new.head()




# Setup the column headers to be used for the final dataframe
skills_list = ['very low skill','low skill','normal skill','high skill','very high skill']

radiant_skills_cols = list(map(lambda s: 'radiant_' + s, skills_list))
dire_skills_cols = list(map(lambda s: 'dire_' + s, skills_list))

radiant_hero_cols = list(map(lambda s: 'radiant_' + s, new_heroes_df['hero_name'].str.lower().values))
dire_hero_cols = list(map(lambda s: 'dire_' + s, new_heroes_df['hero_name'].str.lower().values))



radiant_skills_cols



radiant_hero_cols



match_id_ref = df_merged_new['match_id'].tolist()
match_id_ref_df = pd.DataFrame(match_id_ref,columns=['match_id'])
match_id_ref_df



radiant_skills_dict = {}
dire_skills_dict = {}

for item in radiant_skills_cols:
    count_list = []
    for l in df_merged_new['radiant_skills'].tolist():
        count_list.append(l.count(item))
        
    radiant_skills_dict[item] = count_list

for item in dire_skills_cols:
    count_list = []
    for l in df_merged_new['dire_skills'].tolist():
        count_list.append(l.count(item))
        
    dire_skills_dict[item] = count_list



radiant_skills_dict



radiant_skills_df = pd.DataFrame(radiant_skills_dict)
dire_skills_df = pd.DataFrame(dire_skills_dict)



df_merged_new[df_merged_new['match_id'] == 2]



radiant_hero_dict = {}
dire_hero_dict = {}

for item in radiant_hero_cols:
    count_list = []
    for l in df_merged_new['radiant_heroes'].tolist():
        count_list.append(l.count(item))
        
    radiant_hero_dict[item] = count_list

for item in dire_hero_cols:
    count_list = []
    for l in df_merged_new['dire_heroes'].tolist():
        count_list.append(l.count(item))
        
    dire_hero_dict[item] = count_list


radiant_heroes_df = pd.DataFrame(radiant_hero_dict)
dire_heroes_df = pd.DataFrame(dire_hero_dict)



radiant_heroes_df.head()


df_merged_new[df_merged_new['match_id'] == 0]


radiant_heroes_df['sum'] = radiant_heroes_df.drop('radiant_unknown', axis=1).sum(axis=1)
radiant_heroes_df['radiant_unknown'] = radiant_heroes_df['radiant_unknown'] + (5 -radiant_heroes_df['sum'])

dire_heroes_df['sum'] = dire_heroes_df.drop('dire_unknown', axis=1).sum(axis=1)
dire_heroes_df['dire_unknown'] = dire_heroes_df['dire_unknown'] + (5 -radiant_heroes_df['sum'])

del radiant_heroes_df['sum']
del dire_heroes_df['sum']


skills_heroes_df = pd.concat([match_id_ref_df,
                              radiant_skills_df, 
                              dire_skills_df, 
                              radiant_heroes_df, 
                              dire_heroes_df], axis=1).reset_index(drop=True)
skills_heroes_df



dataframes_final = [df_merged_new[['match_id','region', 'cluster', 'day_of_week','time', 'radiant_win',
                                      'radiant_winrate', 'dire_winrate']
                                 ],
                    skills_heroes_df]
                    

df = reduce(lambda  left,right: pd.merge(left,right,on=['match_id'],
                                            how='outer'), dataframes_final)

df = df.reset_index(drop=True)
df['radiant_win']= df['radiant_win'].astype(int)
del df['match_id']
df



df.isna().sum().sum()


# ## Feature Importance


df['radiant_win'].value_counts(normalize=True)


# Almost same amount of value for radiant and dire wins


categorical = ['region','cluster','day_of_week']
numerical = list(set(list(df.columns.values)).difference(categorical))

categorical.append('time')

numerical.append('time')

print('categorical')
print(categorical)
print()
print('numerical')
print(numerical)



less_categorical = set(list(df.columns.values)).difference(categorical)

l1 = [s for s in less_categorical if "winrate" in s]
less_l1 = set(list(less_categorical)).difference(l1)
l2 = [s for s in less_l1 if "skill" in s]
less_l2 = set(list(less_l1)).difference(l2)

categorical_heroes = less_l2
categorical_skill = l2

numerical = l1
numerical = numerical.append('time')


l1


categorical_skill



categorical_heroes


# ### Risk Ratio



radiant_win_mean = df['radiant_win'].mean()
radiant_win_mean



test = pd.DataFrame()
for c in list(categorical) + list(categorical_skill) + list(categorical_heroes):
 
    df_group = df.groupby(c).radiant_win.agg(['mean', 'count']).reset_index()
    df_group['diff'] = df_group['mean'] - radiant_win_mean
    df_group['risk'] = df_group['mean'] / radiant_win_mean
    df_group['index_new'] = c
    df_group = df_group.set_index('index_new')
    df_group.rename(columns={ df_group.columns[0]: 'item' }, inplace = True)
    
    #display(c)
    #display(df_group)
    
    
    frames = [test, df_group]

    test = pd.concat(frames, axis = 0)



test.sort_values(by='risk',ascending = False)[:20]


test.sort_values(by='risk',ascending = True)[:20]


# ### Mutual Info


def get_mi_score(series):
    return mutual_info_score(series, df['radiant_win'])



mi = df[list(categorical)].apply(get_mi_score)
mi.sort_values(ascending=False)[:20]



mi = df[list(categorical_skill)].apply(get_mi_score)
mi.sort_values(ascending=False)[:20]



mi = df[list(categorical_heroes)].apply(get_mi_score)
mi.sort_values(ascending=False)[:20]


# ### Correlation


df[l1].corrwith(df['radiant_win']).abs()


# No Correlation

# ## Predictive Modelling

# ### Splitting Dataset

df



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)



len(df_train), len(df_val), len(df_test)



df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)



y_train = df_train['radiant_win'].values
y_val = df_val['radiant_win'].values
y_test = df_test['radiant_win'].values

df_train.drop(['radiant_win'], axis=1, inplace = True)
df_val.drop(['radiant_win'], axis=1, inplace = True)
df_test.drop(['radiant_win'], axis=1, inplace = True)



df_train


y_train


y_val


len(df_train), len(y_train),len(df_val), len(y_val),len(df_test), len(y_test)


# ### One Hot Encoding


dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)


# ### Base Modeling


base_score = pd.DataFrame(columns = ['model','score_train', 'score_val'])


# #### Logistic Regression


b_logreg = LogisticRegression(random_state = 1)

b_logreg.fit(X_train, y_train)


# get score on train
y_pred = b_logreg.predict_proba(X_train)[:, 1]
radiant_win_decision = (y_pred >= 0.5)
score_train = roc_auc_score(y_train, y_pred)

# get score on val
y_pred = b_logreg.predict_proba(X_val)[:, 1]
radiant_win_decision = (y_pred >= 0.5)
score_val = roc_auc_score(y_val, y_pred)

# set the key
key = 'logistic regression'


app_list = [key, score_train, score_val]

base_score.loc[len(base_score)] = app_list

base_score


# **Decision Tree**


b_dt = DecisionTreeClassifier(random_state = 1)
b_dt.fit(X_train, y_train)



# get score on train
y_pred = b_dt.predict_proba(X_train)[:, 1]
radiant_win_decision = (y_pred >= 0.5)
score_train = roc_auc_score(y_train, y_pred)

# get score on val
y_pred = b_dt.predict_proba(X_val)[:, 1]
radiant_win_decision = (y_pred >= 0.5)
score_val = roc_auc_score(y_val, y_pred)

# set the key
key = 'decision_tree'


app_list = [key, score_train, score_val]

base_score.loc[len(base_score)] = app_list

base_score


# **Random Forest**


b_rf = RandomForestClassifier(random_state = 1)
b_rf.fit(X_train, y_train)



# get score on train
y_pred = b_rf.predict_proba(X_train)[:, 1]
radiant_win_decision = (y_pred >= 0.5)
score_train = roc_auc_score(y_train, y_pred)

# get score on val
y_pred = b_rf.predict_proba(X_val)[:, 1]
radiant_win_decision = (y_pred >= 0.5)
score_val = roc_auc_score(y_val, y_pred)

# set the key
key = 'random forest'



app_list = [key, score_train, score_val]

base_score.loc[len(base_score)] = app_list

base_score


# **XG Boost**


features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)



xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 10,
    
    'seed': 1,
    'verbosity': 1,
}

b_xgb = xgb.train(xgb_params, dtrain, num_boost_round=10)



# get score on train
#y_pred = model.predict(dtrain)
radiant_win_decision = (y_pred >= 0.5)
score_train = roc_auc_score(y_train, y_pred)

# get score on val
#y_pred = model.predict(dval)
radiant_win_decision = (y_pred >= 0.5)
score_val = roc_auc_score(y_val, y_pred)

# set the key
key = 'xgboost'




app_list = [key, score_train, score_val]

base_score.loc[len(base_score)] = app_list

base_score


# Based from the model, we can see than decision tree and random forest overfits with the training model.
# For the next parts, we will conduct parameter tuning for the 4 models using GridSearchCV

# ### Tuning the Model

# **LogisticRegression**


key = 'c'
params = np.logspace(-4,4,20)
models = pd.DataFrame(columns = [key,'score_train', 'score_val'])

for param in params:
    lr = LogisticRegression(C = param, random_state = 1, n_jobs = 10)
    lr.fit(X_train, y_train)
    
    # get score on train
    y_pred = lr.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred)
    
    # get score on train
    y_pred = lr.predict_proba(X_val)[:, 1]
    score_val = roc_auc_score(y_val, y_pred)
    
    app_list = [param, score_train, score_val]

    models.loc[len(models)] = app_list

models = models.sort_values(by=['score_val'], ascending = False)
models



base_score


# Since, no significant change in score of logistic regression use default c


key = 'max_iter'
params = np.linspace(10, 5000, 50)
models = pd.DataFrame(columns = [key,'score_train', 'score_val'])

for param in params:
    lr = LogisticRegression(max_iter = param, random_state = 1, n_jobs = 10)
    lr.fit(X_train, y_train)
    
    # get score on train
    y_pred = lr.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred)
    
    # get score on train
    y_pred = lr.predict_proba(X_val)[:, 1]
    score_val = roc_auc_score(y_val, y_pred)
    
    app_list = [param, score_train, score_val]

    models.loc[len(models)] = app_list

models = models.sort_values(by=['score_val'], ascending = False)
models


base_score


# no significant change


key = 'penalty'
params = ['l1', 'l2', 'elasticnet', 'none']
models = pd.DataFrame(columns = [key,'score_train', 'score_val'])

for param in params:
    try:
        lr = LogisticRegression(penalty = param, random_state = 1, n_jobs = 10)
        lr.fit(X_train, y_train)

        # get score on train
        y_pred = lr.predict_proba(X_train)[:, 1]
        score_train = roc_auc_score(y_train, y_pred)

        # get score on train
        y_pred = lr.predict_proba(X_val)[:, 1]
        score_val = roc_auc_score(y_val, y_pred)

        app_list = [param, score_train, score_val]

        models.loc[len(models)] = app_list
    except:
        app_list = [param, float('NaN'), float('NaN')]

        models.loc[len(models)] = app_list

models = models.sort_values(by=['score_val'], ascending = False)
models



base_score




key = 'solver'
params = ['lbfgs','newton-cg','liblinear','sag','saga']
models = pd.DataFrame(columns = [key,'score_train', 'score_val'])

for param in params:
    lr = LogisticRegression(solver = param, random_state = 1, n_jobs = 10)
    lr.fit(X_train, y_train)
    
    # get score on train
    y_pred = lr.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred)
    
    # get score on train
    y_pred = lr.predict_proba(X_val)[:, 1]
    score_val = roc_auc_score(y_val, y_pred)
    
    app_list = [param, score_train, score_val]

    models.loc[len(models)] = app_list

models = models.sort_values(by=['score_val'], ascending = False)
models



base_score


# SINCE THERE ARE NO SIGNIFICANT CHANGE FOUND IN THE RESULT. just consider the deault logistic regression


## For checking later
logModel = LogisticRegression()

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000, 2500, 5000]
    }
]

# define search
log_clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=10)


# execute search
best_log_clf = log_clf.fit(X_train, y_train)



best_log_clf.best_estimator_



print (f'Accuracy - : {best_log_clf.score(X_val,y_val):.3f}')


# **DecisionTree**


base_score




dt = DecisionTreeClassifier()

# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

dt_random = RandomizedSearchCV(estimator = dt,
                               param_distributions = params,
                               n_iter = 100,
                               cv = 3,
                               verbose=2,
                               random_state=1,
                               n_jobs = 8)


dt_random.fit(X_train, y_train)



dt_random.best_params_



# get score on train
y_pred = dt_random.predict_proba(X_train)[:, 1]
score_train = roc_auc_score(y_train, y_pred)

# get score on train
y_pred = dt_random.predict_proba(X_val)[:, 1]
score_val = roc_auc_score(y_val, y_pred)

app_list = ['tuned_dt', score_train, score_val]
app_list



base_score


# it improved to 0.55 but still less than the default regression

# ### D.5.1 Random Forest


key = 'n_estimators'
params = range(10, 500, 20)
models = pd.DataFrame(columns = [key,'score_train', 'score_val'])

for param in params:
    rf = RandomForestClassifier(n_estimators = param, random_state = 1, n_jobs = 10)
    rf.fit(X_train, y_train)
    
    # get score on train
    y_pred = lr.predict_proba(X_train)[:, 1]
    score_train = roc_auc_score(y_train, y_pred)
    
    # get score on train
    y_pred = lr.predict_proba(X_val)[:, 1]
    score_val = roc_auc_score(y_val, y_pred)
    
    app_list = [param, score_train, score_val]

    models.loc[len(models)] = app_list

models = models.sort_values(by=['score_val'], ascending = False)
models[:5]



base_score


plt.plot(models.model, models.score_val)


scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1,
                                    n_jobs = -1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))



columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)



for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()



max_depth = 15



scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1,
                                    n_jobs = 6)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))


columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)




colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()



min_samples_leaf = 10



rf = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1,
                            n_jobs=6)
rf.fit(X_train, y_train)


y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)



from sklearn.model_selection import RandomizedSearchCV

# Create the random grid
random_grid = {
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            }



rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 100,
                               cv = 3,
                               verbose=2,
                               random_state=1,
                               n_jobs = 8)



rf_random.fit(X_train, y_train)



rf_random.best_params_



rf = RandomForestClassifier(n_estimators = 600,
                            min_samples_split= 10,
                            min_samples_leaf= 4,
                            max_features='sqrt',
                            max_depth=80,
                            random_state = 1,
                            n_jobs = 10)
rf.fit(X_train, y_train)

# get score on train
y_pred = lr.predict_proba(X_train)[:, 1]
score_train = roc_auc_score(y_train, y_pred)

# get score on train
y_pred = lr.predict_proba(X_val)[:, 1]
score_val = roc_auc_score(y_val, y_pred)



score_train, score_val


# ### D.5.1 XGBoost


features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)



key = 'eta'
params = np.arange(0.1, 0.51, 0.025)
models = pd.DataFrame(columns = [key,'score_train', 'score_val'])

for param in params:
    xgb_params = {
                'eta': param, 
                'max_depth': 6,
                'min_child_weight': 1,

                'objective': 'binary:logistic',
                'nthread': 8,

                'seed': 1,
                'verbosity': 1,
                }
    
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200)
    
    # get score on train
    y_pred = xgb_model.predict(dtrain)
    score_train = roc_auc_score(y_train, y_pred)
    
    # get score on train
    y_pred = xgb_model.predict(dval)
    score_val = roc_auc_score(y_val, y_pred)
    
    app_list = [param, score_train, score_val]

    models.loc[len(models)] = app_list
    

models = models.sort_values(by=['score_val'], ascending = False)



models[:10]


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results



scores = {}
watchlist = [(dtrain, 'train'), (dval, 'val')]



key = 'eta=%s' % (xgb_params['eta'])
#scores[key] = parse_xgb_output(output)
key



for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)

#plt.ylim(0.8, 0.84)
plt.legend()


# Based from the plot, even if 0.1 has the highest scores in the end, we still select 0.15 because it leads 0.1 until around 130 rounds and it's scores is not that far with 0.1


best_eta = 0.15



models = pd.DataFrame(columns = ['max_depth', 'min_child_weight', 'score_train', 'score_val'])

for d in range(1,11):
    for w in range(1,11):
        xgb_params = {
                    'eta': best_eta, 
                    'max_depth': d,
                    'min_child_weight': w,

                    'objective': 'binary:logistic',
                    'nthread': 8,

                    'seed': 1,
                    'verbosity': 1,
                    }

        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200)

        # get score on train
        y_pred = xgb_model.predict(dtrain)
        score_train = roc_auc_score(y_train, y_pred)

        # get score on train
        y_pred = xgb_model.predict(dval)
        score_val = roc_auc_score(y_val, y_pred)

        app_list = [d, w, score_train, score_val]

        models.loc[len(models)] = app_list

models = models.sort_values(by=['score_val'], ascending = False)
models[:5]



tuned_score = pd.DataFrame(columns = [key,'score_train', 'score_val'])

tuned_lr = ['tuned_lr', 0.6523, 0.6365]
tuned_dt = ['tuned_dt', 0.6260, 0.5567]
tuned_rf = ['tuned_dt', 0.6531, 0.6360]
tuned_xgb = ['tuned_xgb', 0.7056, 0.6288]

tuned_score.loc[len(tuned_score)] = tuned_lr
tuned_score.loc[len(tuned_score)] = tuned_dt
tuned_score.loc[len(tuned_score)] = tuned_rf
tuned_score.loc[len(tuned_score)] = tuned_xgb


# ### Selecting the best model


tuned_score



base_score


# We will try the base logistic regression, tuned_dt, and tuned_xgb on the test datasets


models = pd.DataFrame(columns = ['model', 'score_fulltrain', 'score_test'])



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train['radiant_win'].astype(int).values
df_test = df_test.reset_index(drop=True)
y_test = df_test['radiant_win'].astype(int).values

del df_full_train['radiant_win']
del df_test['radiant_win']


dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


mod = 'logistic regression'
lr = LogisticRegression(random_state=1)

lr.fit(X_full_train, y_full_train)

# get score on train
y_pred = lr.predict_proba(X_full_train)[:, 1]
score_fulltrain = roc_auc_score(y_full_train, y_pred)

# get score on train
y_pred = lr.predict_proba(X_test)[:, 1]
score_test = roc_auc_score(y_test, y_pred)

app_list = [mod, score_fulltrain, score_test]

models.loc[len(models)] = app_list



models



dt_random.best_params_

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train['radiant_win'].astype(int).values
df_test = df_test.reset_index(drop=True)
y_test = df_test['radiant_win'].astype(int).values

del df_full_train['radiant_win']
del df_test['radiant_win']


mod = 'Decision Tree'
dt = DecisionTreeClassifier(min_samples_leaf=100, 
                            max_depth=20, 
                            criterion='entropy')

dt.fit(X_full_train, y_full_train)

# get score on train
y_pred = dt.predict_proba(X_full_train)[:, 1]
score_fulltrain = roc_auc_score(y_full_train, y_pred)

# get score on train
y_pred = dt.predict_proba(X_test)[:, 1]
score_test = roc_auc_score(y_test, y_pred)

app_list = [mod, score_fulltrain, score_test]

models.loc[len(models)] = app_list



models


# tuned xgboost: eta = 0.15, max_depth=3, min_child_weight=6


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)



df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train['radiant_win'].astype(int).values

del df_full_train['radiant_win']

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train)
dtest = xgb.DMatrix(X_full_train, label=y_full_train)


mod = 'xgb'

xgb_params = {
    'eta': 0.2, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgmodel = xgb.train(xgb_params, dfulltrain, num_boost_round=200)

# get score on train
#y_pred = model.predict(dfulltrain)
score_fulltrain = roc_auc_score(y_full_train, y_pred)

# get score on train
#y_pred = model.predict(dtest)
score_test = roc_auc_score(y_test, y_pred)

app_list = [mod, score_fulltrain, score_test]

models.loc[len(models)] = app_list


models


# ## Using the chosen model


model = pd.DataFrame(['model','score_train','score_test'])


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train['radiant_win'].astype(int).values
df_test = df_test.reset_index(drop=True)
y_test = df_test['radiant_win'].astype(int).values

del df_full_train['radiant_win']
del df_test['radiant_win']

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.fit_transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train)
dtest = xgb.DMatrix(X_test, label=y_test)


mod = 'xgb'

xgb_params = {
    'eta': 0.2, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgmodel = xgb.train(xgb_params, dfulltrain, num_boost_round=200)


# get score on train
y_pred = xgmodel.predict(dfulltrain)
score_fulltrain = roc_auc_score(y_full_train, y_pred)

# get score on train
y_pred = xgmodel.predict(dtest)
score_test = roc_auc_score(y_test, y_pred)

app_list = [mod, score_fulltrain, score_test]

models.loc[len(models)] = app_list



roc_auc_score(y_test, y_pred)


# ## Saving the chosen model


import bentoml


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train['radiant_win'].astype(int).values

del df_full_train['radiant_win']


dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)



dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train)
dtest = xgb.DMatrix(X_full_train, label=y_full_train)



xgb_params = {
    'eta': 0.2, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgmodel = xgb.train(xgb_params, dfulltrain, num_boost_round=200)




bentoml.xgboost.save_model(
    'dota2_predictor_model',
    xgmodel,
    custom_objects={
        'dictVectorizer': dv
    })

