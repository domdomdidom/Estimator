import pandas as pd
import numpy as np
import sys
import getopt
from pyhive import hive
from datetime import date
from datetime import timedelta
import mysql.connector
from os import listdir
from os.path import isfile, join
import os
from itertools import combinations 
import logging
import imp
import ForecastingEnv
from GBM import GBM

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logger = logging.getLogger(__name__)
pd.set_option("display.max_columns", 2000)
pd.options.display.float_format = '{:.5f}'.format

#### python3 estimator.py --configFile=/opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Estimator/config_estimator.csv --queryFile=/opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Estimator/QMP_Estimator_Configurable.txt --trainingData=/opt/home/abacus/ASA/dev/CTR_Estimation/data

def determine_seperator(filepath):
    if 'csv' in str(filepath)[-5]:
        seperator = ','
    elif 'txt' in str(filepath)[-5]:
        seperator = '\t'
    else:
        return
    return seperator

def get_original_shares(df, config_dict):
    '''
    obtain the original distribution of impressions, clicks, rank & matched searches grouped in the form of output_df
    '''

    for col in config_dict['attributes']:
        df[col].fillna('Unknown', inplace=True)

    shown = df[df['advertiser.accountid'] == config_dict['accnt_id']]
    shown['spend'] = shown['clickoccurred'] * shown['bid']

    output_df = pd.DataFrame(columns = ['trafficday', 'qmp_profile', 'state_code', 'devicetype', 'requesttype','publisher.company','publisher.accountname','publisher.accountid','encrypted_affiliate_key', 'attribute', 'attribute_level', 'matched_searches', 'original_impressions', 'original_clicks', 'original_rank', 'original_spend', 'original_avg_bid'])

    for attribute in config_dict['attributes']:

        gb_cols = ['trafficday','publisher.company','publisher.accountname',
            'publisher.accountid','encrypted_affiliate_key', 'requesttype',
            'qmp_profile', 'state_code', 'devicetype', attribute]

        gb1 = df.fillna(0).groupby(gb_cols).agg(
            total_clicks_on_segment=('clickoccurred',np.sum), matched_searches=('searchid',pd.Series.nunique)).reset_index().rename(columns={attribute : 'attribute_level'})

        gb2 = shown.fillna(0).groupby(gb_cols).agg(
            original_impressions=('impressionid', np.size), original_clicks=('clickoccurred', np.sum),
            original_rank=('position', np.mean), original_spend=('spend', np.sum),
            original_avg_bid=('bid', np.mean)).reset_index().rename(columns={attribute : 'attribute_level'})

        gb = pd.merge(gb1, gb2, how='left', on=['trafficday','publisher.company','publisher.accountname','publisher.accountid','encrypted_affiliate_key', 'requesttype', 'qmp_profile', 'state_code', 'devicetype', 'attribute_level'])

        gb['attribute'] = attribute

        output_df = output_df.append(gb)

    return output_df     
              
def get_potential_listings(df, config_dict):
    '''
    label each impression as TRUE/FALSE
    '''
    searches_appeared_in = np.unique(df[df['advertiser.accountid'] == config_dict['accnt_id']]['searchid'])
    potential_listings = set(np.unique(df['searchid'])) ^ set(searches_appeared_in)

    df['is_potential_listing'] = df['searchid'].isin(potential_listings)
    pl_df = df[df['is_potential_listing'] ==  True]

    return df

def compute_bids(df, config_dict):
    '''
    compute new bids using click and phone bid from impsup table
    label all impressions with impressiontype_id (needed for gbm)
    '''

    # appears in listing already
    appeared = df[df['advertiser.accountid'] == config_dict['accnt_id']][['searchid', 'bid', 'impression.type']].set_index('searchid')

    # did not appear in listing
    potential = df[df['is_potential_listing'] == True][['searchid', 'impsup_target_cb', 'impsup_target_pb', 'devtype_modifier']].drop_duplicates().set_index('searchid')
    
    # calculate bids with click_bid, phone_bid
    potential['bid'] = potential['impsup_target_cb'] + (potential['devtype_modifier']/100. * potential['impsup_target_pb'])

    # create column for impression type
    conditions = [
        ((potential['impsup_target_cb'] > 0) & (potential['impsup_target_pb'] > 0)),
        ((potential['impsup_target_cb'] > 0) & (potential['impsup_target_pb'] == 0)),
        ((potential['impsup_target_cb']) == 0 & (potential['impsup_target_pb'] > 0))]

    choices = [5, 4, 8]

    potential['impression.type'] = np.select(conditions, choices, default=4)

    # overwrite TIB column with new bids
    target_data = appeared.append(potential[['bid', 'impression.type']]).rename(columns={'bid':'target_initial_bid', 'impression.type':'target_impression.type'})

    return pd.merge(df, target_data, how='left', left_on='searchid', right_index=True)

def create_new_bid(df, multiplier):
    '''
    use the multiplier to calculate the new bid
    '''
    df['new_bid'] = df['target_initial_bid'] * float(multiplier)

def calculate_rpi(gbm_model, df, config_dict):
    
    gbm_variables = ['military_affiliation', 'insurance_carrier', 'gender', 'stateid',  'devicetype',
                'listing_style_id', 'publisher.accountid', 'publisher.customerid', 'target_impression.type',
                'advertiser.customerid', 'advertiser.accountid', 'position', 'listingset', 'impressionid', 'searchid']

    gbm_df = df[df['displaystrategyname'].str.contains('AUTO-GBM')]
    gbm_df_with_gbm_variables = gbm_df[gbm_variables].rename(columns={'target_impression.type' : 'impression.type'})
    gbm_df_with_gbm_variables['advertiser.customerid'], gbm_df_with_gbm_variables['advertiser.accountid'] = config_dict['cust_id'], config_dict['accnt_id']
    gbm_df_with_gbm_variables['listing_style_id'] = gbm_df_with_gbm_variables['listing_style_id'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else x)
    gbm_df_with_gbm_variables = gbm_df_with_gbm_variables.astype(str).replace({'na': 'nan'}).replace({'NaN': 'nan'}).replace({'': 'nan'}).replace({'Unknown': 'nan'}).set_index('impressionid')

    ### run the GBM

    gbm_df_with_variables_reset_index = gbm_df_with_gbm_variables.reset_index()
    result = gbm_model.getProbabilityDF(gbm_df_with_variables_reset_index, 'impressionid').set_index('impressionid')
    
    result.index = result.index.astype(int)
    gbm_df['impressionid'] = gbm_df['impressionid'].astype(int)

    ### merge the results back to gbm_df, get client_est_rpi

    ctr_at_position = pd.merge(gbm_df, result, left_on='impressionid' , right_index=True)
    ctr_at_position['client_est_rpi'] = ctr_at_position['Prediction'] * ctr_at_position['new_bid']

    frames = [df[~df['displaystrategyname'].str.contains('AUTO-GBM')], ctr_at_position]
    return pd.concat(frames)
    
def adjust_bid(config_dict, multiplier, df, gbm_model):

    multiplier_df = df

    create_new_bid(multiplier_df, multiplier) # creates a column for the new bid
    rpi_adjusted = calculate_rpi(gbm_model, multiplier_df, config_dict)
    create_outrank(rpi_adjusted, multiplier, config_dict) # see if new bid > orig bid
    create_passed(rpi_adjusted) # see if the new bid makes it into the listing

    return rpi_adjusted

def create_outrank(df, multiplier, config_dict):
    '''
    determine who client has outranked. different choices for multiplier 1.0 so they "outrank themselves"
    '''
    df[['client_est_rpi', 'rpi', 'minrpi_inlisting']] = df[['client_est_rpi', 'rpi', 'minrpi_inlisting']].apply(pd.to_numeric, errors='coerce')

    conditions = [
        ( (~df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['new_bid'] > df['bid']) ),
        ( (~df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['new_bid'] == df['bid']) & (df['advertiser.accountid'] == config_dict['accnt_id']) ),
        ( (~df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['new_bid'] == df['bid']) & (df['advertiser.accountid'] != config_dict['accnt_id']) ),
        ( (~df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['new_bid'] < df['bid']) ), 
        
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['rpi_pos_mismatch_flag'] == 1) & (df['client_est_rpi'] > (config_dict['rank_pos_mismatch_thresh']*df['rpi'])) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['rpi_pos_mismatch_flag'] == 1) & (df['advertiser.accountid'] == config_dict['accnt_id']) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['rpi_pos_mismatch_flag'] == 1) & (df['client_est_rpi'] == (config_dict['rank_pos_mismatch_thresh']*df['rpi'])) & (df['advertiser.accountid'] != config_dict['accnt_id']) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['rpi_pos_mismatch_flag'] == 1) & (df['client_est_rpi'] < (config_dict['rank_pos_mismatch_thresh']*df['rpi'])) ),


        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['listingset'] != 1) & (df['rpi_pos_mismatch_flag'] == 0) & (df['client_est_rpi'] > df['rpi']) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['listingset'] == 1) & (df['rpi_pos_mismatch_flag'] == 0) & (df['client_est_rpi'] > (config_dict['rank_pos_mismatch_thresh']*df['rpi'])) ), # more frequent 2A scenarios happen with small listingsets, this is forcing them into a 2B scenario

        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['rpi_pos_mismatch_flag'] == 0) & (df['client_est_rpi'] == df['rpi']) & (df['advertiser.accountid'] == config_dict['accnt_id']) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['rpi_pos_mismatch_flag'] == 0) & (df['client_est_rpi'] == df['rpi']) & (df['advertiser.accountid'] != config_dict['accnt_id']) ),

        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['listingset'] != 1) & (df['rpi_pos_mismatch_flag'] == 0) & (df['client_est_rpi'] < df['rpi']) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['listingset'] == 1) & (df['rpi_pos_mismatch_flag'] == 0) & (df['client_est_rpi'] < (config_dict['rank_pos_mismatch_thresh']*df['rpi'])) )
    ]

    choices = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    choices_mod1 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]

    if multiplier == 1.0: 
        choices = choices_mod1
    
    df['outrank'] = np.select(conditions, choices)


def create_passed(df):
    '''
    see if new bid / new rpi exceeds final_min_bid / minrpi_inlisting
    '''
    conditions = [
        ( (~df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['new_bid'] > df['final_min_bid']) ),
        ( (~df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['new_bid'] <= df['final_min_bid']) ),
        
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['client_est_rpi'] > df['minrpi_inlisting']) ),
        ( (df['displaystrategyname'].str.contains('AUTO-GBM')) & (df['client_est_rpi'] <= df['minrpi_inlisting']) )
    ]
    choices = [1, 0, 1, 0]
    
    df['passed'] = np.select(conditions, choices, default=0)

def filter_data(df):
    '''
    filter to "made_it_in" (where they now have entered the listing)
    append these new searches to "already_there" searches
    '''
    already_there = df[df['is_potential_listing']  == False]

    made_it_in_searches = np.unique(df[(df['is_potential_listing'] == True) & (df['passed'] == 1)]['searchid']) 

    return already_there.append(df[df['searchid'].isin(made_it_in_searches)])

def rerank(df, multiplier, config_dict):
    '''
    calculate new position of client using outrank column...corresponds with toyExample on Jira
    also identifies problem searches, where they should have made it into the listing originally and did not for whatever reason (these output as csv with inspect_searches=True)
    '''
    
    df['calc_1'] = (-1) / (df['outrank'] * df['position'])
    df['calc_2'] = df['calc_1'].replace(-np.inf, 0) 

    grp = df.groupby(['searchid', 'is_potential_listing']).agg({'outrank':np.sum, 'calc_2':np.min, 'listingset':np.min}).reset_index().set_index('searchid')

    conditions = [
        ( (grp['is_potential_listing'] == True) & (grp['outrank'] == 0) ),
        ( (grp['is_potential_listing'] == False) & (grp['outrank'] == 0) ),
        ( (grp['is_potential_listing'] == True) & (grp['outrank'] > 0) ),
        ( (grp['is_potential_listing'] == False) & (grp['outrank'] > 0) ),
        ]

    choices = [grp['listingset']+1, grp['listingset'], -1/grp['calc_2'], -1/grp['calc_2']]

    grp['new_pos'] = np.select(conditions, choices)

    combined_ranks = determine_scenario(df, multiplier, grp['new_pos'], config_dict)

    # drop the 2B cases from GBM
    drop_cases = list(set(combined_ranks[(combined_ranks['displaystrategyname'].str.contains('AUTO-GBM')) & (combined_ranks['scenario'] == '2B')].index))
    df, combined_ranks = df[~df['searchid'].isin(drop_cases)], combined_ranks[~combined_ranks.index.isin(drop_cases)]

    # check if the client est rpi doesn't match what we have in the db
    if multiplier == 1.0:
        mismatch = df[(df['advertiser.accountid'] == config_dict['accnt_id']) & (df['displaystrategyname'].str.contains('AUTO-GBM')) & (round(df['client_est_rpi'],0) != round(df['rpi'],0))] [['searchid', 'displaystrategyname', 'rpi', 'client_est_rpi']]
        logger.info('%s rows do not have estimated ctr = ctr from bigdata' % str(len(mismatch)))
        logger.info(mismatch)

    return combined_ranks, df

def determine_scenario(df, multiplier, new_ranks, config_dict):
    '''
    1A - rank improved
    1B - rank did not change
    1C - rank got lower
    2A - made it in listing, new rank is pre-existing rank
    2B - made it in listing, added to end of listing
    '''

    old_ranks = df[df['advertiser.accountid'] == config_dict['accnt_id']][['searchid', 'position']].drop_duplicates().set_index('searchid')

    combined_ranks = pd.merge(new_ranks, old_ranks,
                    how = 'left',
                    left_index=True,
                    right_index=True).rename(columns={'position' : 'old_pos'})

    combined_ranks = pd.merge(combined_ranks, df[['searchid', 'listingset', 'displaystrategyname', 'rpi_pos_mismatch_flag']].drop_duplicates(), 
                        left_index=True, right_on='searchid')

    conditions = [
        (combined_ranks['new_pos'] < combined_ranks['old_pos']),
        (combined_ranks['new_pos'] == combined_ranks['old_pos']),
        (combined_ranks['new_pos'] > combined_ranks['old_pos']),
        (combined_ranks['old_pos'].isna() & (combined_ranks['new_pos'] <= combined_ranks['listingset'])),
        (combined_ranks['old_pos'].isna() & (combined_ranks['new_pos'] > combined_ranks['listingset']))]

    choices = ['1A', '1B', '1C', '2A', '2B']
    
    combined_ranks['scenario'] = np.select(conditions, choices, default='2B')

    # extend the listingset by 1 for scenarios 2A and 2B, when the original listing set was less than 8
    combined_ranks.loc[(combined_ranks['scenario'] == '2B') & (combined_ranks['listingset'] < 8), 'listingset'] += 1
    combined_ranks.loc[(combined_ranks['scenario'] == '2A') & (combined_ranks['listingset'] < 8), 'listingset'] += 1

    # IDENTIFY PROBLEM SEARCHES    
    if multiplier == 1.0:
        problem_searches = combined_ranks[combined_ranks['scenario'] != '1B']['searchid']
        problem_search_df = df[df['searchid'].isin(problem_searches)][['searchid', 'displaystrategyname', 'listing_delivery_type_id', 'phonecallpub', 'publisher.accountname', 'advertiser.accountname', 'target_client_shown_flag', 'suppressionreason','final_min_bid', 'target_client_shown_bid', 'bid', 'target_initial_bid', 'new_bid', 'passed', 'outrank']]
        problem_search_df.to_csv("problem_searches_inspection.csv", index=False)

    return combined_ranks.set_index('searchid')

def format_GBM(df, venki_data, gbm_model):

    gbm_variables = list(gbm_model.variablesDictByName.keys())[1:]
    changing_cols = ['impression.type', 'advertiser.customerid', 'advertiser.accountid', 'position', 'listingset']
    attribute_cols = [col for col in gbm_variables if col not in changing_cols]
    attribute_cols.insert(0, 'searchid')
    filtered_venki = venki_data[attribute_cols].drop_duplicates()
    filtered_venki['searchid'] = filtered_venki['searchid'].astype(int)
    df['searchid'] = df['searchid'].astype(int)

    gbm_ready = pd.merge(filtered_venki, df, how='inner',
                        left_on = 'searchid', 
                        right_on = 'searchid').set_index('searchid')

    logger.info('%s searches could not be estimated by the GBM' % str(len(df) - len(gbm_ready)))
    logger.info('%s searches can be estimated using the GBM' % str(len(gbm_ready)))

    gbm_ready['listing_style_id'] = gbm_ready['listing_style_id'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else x)
    return gbm_ready[gbm_variables].astype(str).replace({'na': 'nan'}).replace({'NaN': 'nan'}).replace({'': 'nan'}).replace({'Unknown': 'nan'})

def add_clicks(df, combined_ranks, old_pos_clicks=True):

    # orig_clicks
    orig_clicks = df[df['clickoccurred'] > 0].groupby(['searchid','position']).size().to_frame()

    # add orig clicks in new pos
    mrg = pd.merge(combined_ranks.reset_index(), orig_clicks.reset_index(),  
                     how='left',
                     left_on = ['searchid','new_pos'], 
                     right_on = ['searchid','position']).set_index('searchid').drop(columns=['position']).rename(columns={0 : 'original_clicks_new_pos'})
        
    # add orig clicks in old pos
    mrg = pd.merge(mrg, orig_clicks.reset_index(),  
                    how='left',
                    left_on = ['searchid','old_pos'], 
                    right_on = ['searchid','position']).set_index('searchid').drop(columns=['position']).rename(columns={0 : 'original_clicks_old_pos'})
    
    # zero out clicks from old_pos when 1B, otherwise they are double counted (because rank did not change in 1B)
    mrg.loc[mrg['old_pos'] == mrg['new_pos'], 'original_clicks_old_pos'] = 0

    # zero out clicks from new_pos when 1C, forcing them to keep their higher rank click
    mrg.loc[mrg['scenario'] == '1C', 'original_clicks_new_pos'] = 0

    if old_pos_clicks==True:
        mrg['original_clicks'] = mrg['original_clicks_new_pos'].fillna(0) + mrg['original_clicks_old_pos'].fillna(0)
    else:
        mrg['original_clicks'] = mrg['original_clicks_new_pos'].fillna(0)
    
    # set the max clicks per search to be 1...otherwise if they move from rank 2 to rank 1 and there were clicks in both, they'll get 2 clicks
    mrg.loc[mrg['original_clicks'] > 1, 'original_clicks'] = 1

    return mrg

def format_newranks(df, combined_ranks, data_modified, multiplier, venki_data, config_dict, gbm_model, old_pos_clicks=True):
    '''
    get the new positions join to clicks from old/new positions
    extend listingsets where necessary
    for end of listing cases, trunc dataframe and format for GBM 
    '''
    
    # add clicks, extend ls where appropriate
    clicks_added = add_clicks(data_modified, combined_ranks)

    # add impression.type
    all_cols_added = pd.merge(clicks_added, df[['searchid', 'target_impression.type']].drop_duplicates(),
                     left_on = 'searchid', right_on = 'searchid')
        
    # format case_2 for GBM
    case_2 = all_cols_added[all_cols_added['scenario'] == '2B']
    case_2['advertiser.customerid'], case_2['advertiser.accountid'] = config_dict['cust_id'], config_dict['accnt_id']
    case_2.rename(columns={'new_pos':'position', 'target_impression.type':'impression.type'}, inplace=True)

    case_2_GBM_formatted = format_GBM(case_2, venki_data, gbm_model)

    return all_cols_added, case_2_GBM_formatted

def format_output(original, output_df, config_dict):

    original_exploded = pd.DataFrame()

    for multiplier in config_dict['multipliers']:

        original['multiplier'] = multiplier
        original_exploded = original_exploded.append(original)

        out = pd.merge(original_exploded, output_df, how='left',
                    left_on = ['trafficday', 'publisher.accountid', 'publisher.accountname', 'encrypted_affiliate_key', 'requesttype', 'attribute', 'attribute_level', 'qmp_profile', 'state_code', 'devicetype', 'multiplier'], 
                    right_on = ['trafficday', 'publisher.accountid', 'publisher.accountname', 'encrypted_affiliate_key', 'requesttype', 'attribute', 'attribute_level', 'qmp_profile', 'state_code', 'devicetype', 'multiplier'])

    out['gbm_valid_pub'] = out['publisher.accountid'].apply(lambda x: 1 if x in config_dict['gbm_valid_pubs'] else 0)

    out.loc[out['gbm_valid_pub'] == 0, ['estimated_clicks', 'estimated_spend']] = np.nan

    out['advertiser.customerid'], out['advertiser.company'] = config_dict['cust_id'], config_dict['advertiser.company']
    out = out[['trafficday', 'requesttype', 'qmp_profile', 'state_code', 'devicetype', 'advertiser.company', 'advertiser.customerid', 'encrypted_affiliate_key', 'gbm_valid_pub', 'publisher.company','publisher.accountname','publisher.accountid', 'algo_type', 'total_clicks_on_segment', 'multiplier', 'attribute', 'attribute_level', 'matched_searches', 'original_impressions', 'estimated_impressions', 'original_clicks', 'estimated_clicks', 'original_rank', 'estimated_rank', 'original_spend', 'estimated_spend', 'original_avg_bid', 'estimated_avg_bid']]

    return out

def parse_config(params_dict):
    
    logger.info("START: readConfig")
    n_errors = 0
    configFile = params_dict['configFile']
    
    try:
        config = pd.read_csv(configFile, sep = determine_seperator(configFile), engine='python')

        if not isinstance(pd.read_csv(configFile, sep = determine_seperator(configFile), engine='python'), pd.core.frame.DataFrame):
            logger.error("readConfig: csv/txt cannot be read as pandas dataframe")
            n_errors += 1

        config.columns = [x.upper() for x in config.columns] # make everything uppercase

        target_cols = ['RANK_POS_MISMATCH_THRESH','ACCOUNT_ID','CUSTOMER_ID','VENDOR_KEY','ATTRIBUTES','MULTIPLIERS','DATE_COL_NAME','N_DAYS','GBM_VERSION']

        # Check Validity of Config File
        for col in config.columns:
            if col not in target_cols:
                logger.warning("readConfig: Misspelled or extra column in config file: " + str(col) + str('\n') + str("Valid Column Names (Not Case Sensitive): 'ACCOUNT_ID','CUSTOMER_ID','VENDOR_KEY','ATTRIBUTES','MULTIPLIERS','DATE_COL_NAME','N_DAYS','GBM_VERSION', RANK_POS_MISMATCH_THRESH]"))
        
        for col in target_cols:
            if col not in config.columns:
                logger.error("readConfig: Missing column from config file: " + str(col) + str('\n') + str("Valid Column Names (Not Case Sensitive): 'ACCOUNT_ID','CUSTOMER_ID','VENDOR_KEY','ATTRIBUTES','MULTIPLIERS','DATE_COL_NAME','N_DAYS','GBM_VERSION', RANK_POS_MISMATCH_THRESH]"))
                n_errors += 1

        return n_errors, config

    except:
        logger.error("readConfig: config file not found at " + str(configFile))
        n_errors += 1

def zip_client_config_dict(i, config):

    config_dict = {'attributes' : [item.strip() for item in config['ATTRIBUTES'][i].split(',')],
                'multipliers' : [item.strip() for item in config['MULTIPLIERS'][i].split(',')],
                'gbm_version' : str('/opt/home/abacus/ASA/AutoRun_CTR/workspace/gbm_models/' + config['GBM_VERSION'][i]),
                'accnt_id' : config['ACCOUNT_ID'][i],
                'vendor_key' : config['VENDOR_KEY'][i],
                'cust_id' : config['CUSTOMER_ID'][i],
                'date_col_name' : str(config['DATE_COL_NAME'][i]),
                'n_days' : config['N_DAYS'][i],
                'rank_pos_mismatch_thresh' : config['RANK_POS_MISMATCH_THRESH'][i]}

    config_dict['t1'] = ((date.today() - timedelta(1)) - timedelta(float(config_dict['n_days'])-1))
    config_dict['t2'] = (date.today() - timedelta(1))

    logger.info("readConfig: ACCOUNT_ID read from config as %s " % config_dict['accnt_id'])
    logger.info("readConfig: CUSTOMER_ID read from config as %s " % config_dict['cust_id'])
    logger.info("readConfig: VENDOR_KEY read from config as %s " % config_dict['vendor_key'])
    logger.info("readConfig: GBM_VERSION read from config as %s " % config_dict['gbm_version'])
    logger.info("readConfig: DATE_COL_NAME read from config as  %s " % config_dict['date_col_name'])
    logger.info("readConfig: ATTRIBUTES read from config as %s " % config_dict['attributes'])
    logger.info("readConfig: MULTIPLIERS read from config as %s " % config_dict['multipliers'])
    logger.info("readConfig: N_DAYS read from config as %s " % config_dict['n_days'])
    logger.info("readConfig: RANK_POS_MISMATCH_THRESH read from config as %s " % config_dict['rank_pos_mismatch_thresh'])
    logger.info("FINISHED: readConfig")

    return config_dict

def add_EOL_case(df):
    
    search_level = df.set_index('searchid').groupby(df.index).first()
    
    eol_cases = df[df['listingset'] < 8][['searchid', 'listingset', 'position']].groupby('searchid').max().apply(lambda x: x+1)
    
    merged = pd.merge(eol_cases, search_level, 
                        left_index=True, right_index=True, how='left')
    
    concatenated = df.append(merged)
    
    return concatenated

def read_venki(params_dict, config_dict, gbm_model):

    logger.info("Beginning read gbm training files")

    date_list = [(config_dict['t1'] + timedelta(days = i)).isoformat() for i in range((config_dict['t2']-config_dict['t1']).days+1)]

    onlyfiles = [f for f in listdir(params_dict['trainingData']) if isfile(join(params_dict['trainingData'], f))]

    initial_listings_where = [',' + str(config_dict['accnt_id']) + '=' , str(config_dict['accnt_id']) + '=']

    n_days_data = pd.DataFrame()

    for f in onlyfiles:
        for dt in date_list:
            if dt in f:
                logger.info('ReadGBMTraining: Reading and filtering for %s ', dt)

                day_data = pd.read_csv(params_dict['trainingData']+'/'+str(f), sep=determine_seperator(params_dict['trainingData']+'/'+str(f)), engine='python')
                day_data['trafficday'] = dt
                day_data['filtered_data.initial_listings'] = day_data['filtered_data.initial_listings'].fillna('0')

                # append the filtered data for a single day
                n_days_data = n_days_data.append(day_data[day_data['filtered_data.initial_listings'].str.contains('|'.join(initial_listings_where))])
            else:
                pass            

    check_search_level_attributes(gbm_model, n_days_data) 

    config_dict['gbm_valid_pubs'] = set(n_days_data['publisher.accountid'])

    return n_days_data

def check_search_level_attributes(gbm_model, venki_data):

    gbm_variables = list(gbm_model.variablesDictByName.keys())[1:]
    changing_cols = ['impression.type', 'advertiser.customerid', 'advertiser.accountid', 'position', 'listingset']
    n_searches = len(set(venki_data['searchid']))

    for variable in gbm_variables:
        if variable in changing_cols:
            pass
        else:
            if n_searches != len(venki_data[['searchid', variable]].fillna('Unknown').groupby(['searchid', variable]).count()):
                logger.warning('WARNING: Variable inconsistent accross search level - %s', variable)
            else:
                logger.info('Variable search level consistency - PASS - %s', variable)
    
def pull_hive_data(params_dict, config_dict):

    n_errors = 0

    logger.info("Pulling additional fields from Hive")

    queryFile = open(str(params_dict['queryFile']), "r")
    query_stmt = queryFile.read().replace('ACCOUNT_ID' , str(config_dict['accnt_id'])).replace('VENDOR_KEY' , str(config_dict['vendor_key'])).replace('DATE_1' , "'"+str(config_dict['t1'].isoformat())+"'").replace('DATE_2' , "'"+str(config_dict['t2'].isoformat())+"'").replace('DATE_SL_1' , "'"+str((config_dict['t1'] - timedelta(1)).isoformat())+"'").replace('DATE_SL_2' , "'"+str((config_dict['t2'] + timedelta(1)).isoformat())+"'")
    queryFile.close()

    hive_conn = hive.connect(host=ForecastingEnv.BIGDATA_SERVER_STRING, 
                              port=ForecastingEnv.BIGDATA_SERVER_PORT,  
                              username=ForecastingEnv.BIGDATA_USER, 
                              auth='LDAP', 
                              password=ForecastingEnv.BIGDATA_PASSWORD,
                              database='default')
    hive_curs = hive_conn.cursor()

    try: 
        hive_data = pd.read_sql(query_stmt, hive_conn)
        logger.info("readData: query executed successfully. Returned dataframe of row length = " + str(len(hive_data)))
        hive_curs.close()
        hive_conn.close()

        if len(hive_data) <= 1:
            logger.error("readData: empty Dataframe")
            n_errors += 1

        try:
            # ensure date column in proper format (name, dtype)
            hive_data.rename(columns={config_dict['date_col_name']:'trafficday'}, inplace=True)
            hive_data['trafficday'] = pd.to_datetime(hive_data['trafficday'])

        except:
            logger.error("readData: DATE COL in data file invalid date format or does not match DATE_COL_NAME in config: " + config_dict['date_col_name'])
            n_errors += 1

    except:
        logger.error("readData: query failure")
        n_errors += 1
        hive_curs.close()
        hive_conn.close()

    return hive_data, n_errors
  
def pushDataToDB(dataframe, columns, initial_statement, insert_statement, updateBatchSize = 500):
    conn = mysql.connector.connect(host = ForecastingEnv.BD_MYSQL_HOST
                                                , database = "bd_reports"
                                                , port = ForecastingEnv.BD_MYSQL_PORT
                                                , user = ForecastingEnv.BD_MYSQL_USER
                                                , password = ForecastingEnv.BD_MYSQL_PASSWORD, use_pure = True)
    curs = conn.cursor()
    logger.info("Established connection to MySQL")

    try:
        logger.info("START: pushDataToDB")

        logger.info("Prepare data for update")
        updatesList = dataframe.loc[:,columns].to_records(index=False).tolist()

        logger.info( "Executing initial statement")
        logger.info(initial_statement)
        curs.execute(initial_statement)
        logger.info("Finished executing initial statement")

        elementCount = len(updatesList)
        logger.info("Will insert %d rows", elementCount)
        logger.info(insert_statement)
        if updateBatchSize==0:
            updateBatchSize=elementCount
        startIndex = 0
        endIndex = min(startIndex + updateBatchSize,elementCount)
        while(startIndex < elementCount):
            rows = updatesList[startIndex:endIndex][0]

            logger.info(" Rows " + str(rows))
            curs.execute(insert_statement, rows)  

            startIndex = endIndex #CHANGING THIS USED TO BE ENDINDEX + 1
            endIndex = min(startIndex + updateBatchSize, elementCount)
        logger.info("Finished insertion of %d rows", elementCount)

        logger.info("Finished updating DB")
        conn.commit()

        logger.info("Closing DB connection")
        curs.close()
        conn.close()

    except TypeError as error:
                logger.error("TypeError: %s", error)

    except mysql.connector.Error as error:
                logger.error("Error while connecting to (MYSQL): %s", error)
                if conn != None:
                        if curs != None:
                                curs.close()
                        conn.close()
                raise
    except:
            logger.error("Exception neither Oracle nor MYSQL", sys.exc_info()[0])
            if conn != None:
                    if curs != None:
                            curs.close()
                    conn.close()
            raise

def estimator(data, config_dict, params_dict):

    config_dict['advertiser.company'] = data[data['advertiser.accountid'] == config_dict['accnt_id']].head(1)['advertiser.company'].values[0]

    gbm_model = GBM()
    gbm_model.readFromPath(config_dict['gbm_version'])

    # groupbys on attribute, orginal distribution
    original = get_original_shares(data, config_dict)

    data_modified = get_potential_listings(data, config_dict) 
    data_modified = compute_bids(data_modified, config_dict)
    venki_data = read_venki(params_dict, config_dict, gbm_model)

    output_df = pd.DataFrame(columns = ['trafficday', 'publisher.accountid', 'publisher.accountname', 
                                        'gbm_valid_pub', 'encrypted_affiliate_key', 'qmp_profile', 
                                        'state_code', 'devicetype', 'requesttype', 'algo_type', 
                                        'multiplier', 'attribute', 'attribute_level', 'estimated_impressions', 
                                        'estimated_clicks', 'estimated_rank', 'estimated_spend', 'estimated_avg_bid'])

    for multiplier in config_dict['multipliers']:

        logger.info('Estimator: computing bid adjustments for multiplier: ' + str(multiplier))

        bid_adjusted = adjust_bid(config_dict, multiplier, data_modified, gbm_model) # create cols 'new_bid', 'passed', 'outrank'
        made_it_in = filter_data(bid_adjusted) # shows only where 'passed' == 1
        combined_ranks, made_it_in = rerank(made_it_in, multiplier, config_dict) # account for edge cases, where we need to extend listing sets
    
        logger.info('Estimator: formatting outputs for GBM')

        # run GBM on "type 2" searches to estimate clicks
        pos_out, case_2_GBM_formatted = format_newranks(made_it_in, combined_ranks, data_modified, multiplier, venki_data, config_dict, gbm_model, old_pos_clicks=True)
        logger.info('Estimator: starting GBM for multiplier: ' + str(multiplier))

        for df, name in zip([case_2_GBM_formatted], ['stolen_pos']):
            
            logger.info('GBM:  algo type: ' + str(name))

            df = df.reset_index()

            result = gbm_model.getProbabilityDF(df, 'searchid').set_index('searchid')

            #merge gbm result back to pos_out
            result1 = pd.merge(pos_out.set_index('searchid'), result, left_index=True, right_index=True, how='outer') # merge GBM estimated with stolen POS
            
            result1['Prediction_Combined'] = result1['Prediction'].fillna(0) + result1['original_clicks'].fillna(0)
            result1['click.source'] =  result1['scenario'].apply(lambda x: 'GBM' if x == '2B' else 'POS')  
            result1['position'] = result1['new_pos'] # use the new positions

            result_w_bids = pd.merge(result1, made_it_in.groupby('searchid')['new_bid'].max().to_frame(), left_index=True, right_index=True, how='left')

            df = pd.merge(result_w_bids, data[["trafficday", "publisher.accountid", "publisher.accountname", "encrypted_affiliate_key", "state_code", "devicetype", "qmp_profile", "searchid","agebracket","vehicle_count","currently_insured","home_owner", "requesttype"]].drop_duplicates(), left_index=True, right_on='searchid', how='left')

            df['spend'] = df['Prediction_Combined'] * df['new_bid'] 

            for attribute in config_dict['attributes']:    

                gb = df.fillna(0).groupby(['trafficday', 'publisher.accountid', 'publisher.accountname', 'encrypted_affiliate_key', 
                    'qmp_profile','state_code', 'devicetype','requesttype', attribute]).agg(
                    estimated_impressions=('position', np.size), estimated_clicks=('Prediction_Combined', np.sum),
                    estimated_rank=('position', np.mean), estimated_spend=('spend', np.sum),
                    estimated_avg_bid=('new_bid', np.mean)).reset_index().rename(columns={attribute : 'attribute_level'})

                gb['multiplier'], gb['attribute'], gb['algo_type']  = multiplier, attribute, name

                output_df = output_df.append(gb)

        logger.info('Estimator: completed for multiplier: ' + str(multiplier))

    logger.info(output_df)

    final = format_output(original, output_df, config_dict)
    
    return final
    
def main(argv):

    params_dict={"configFile":"", "queryFile":"", "trainingData":""}

    try:
        params_read, args = getopt.getopt(argv,"h",[param+"=" for param in params_dict])

    except getopt.GetoptError as e:
        logger.info("getopt.GetoptError=%s",e)
        sys.exit(2)
    for param, value in params_read:
        if param == '-h':
            sys.exit()
        elif param[2:] in params_dict:
            params_dict[param[2:]] = value
    for param, value in params_dict.items():
        if (value==""):
            logger.info("Missing parameter '%s'",param)
            sys.exit(2)

    n_errors, config = parse_config(params_dict)
    if n_errors > 0:
        logger.error("Breaking on error in parse_config") 
        return

    all_client_dataframe = pd.DataFrame()

    for i in range(0, len(config)):

        config_dict = zip_client_config_dict(i, config)

        data, n_errors = pull_hive_data(params_dict, config_dict)

        if n_errors > 0:
            logger.error("Breaking on error in pull_data from Hive") 
            return
        try:
            single_client_estim_output = estimator(data, config_dict, params_dict)
            logger.info("Completed for %s", config_dict['advertiser.company']) 
            all_client_dataframe = all_client_dataframe.append(single_client_estim_output)
        except:
            logger.info("Failed on client - %s", config_dict['accnt_id'])
            pass

    all_client_dataframe.to_csv('estim_algo_output.csv', index=False)
    logger.info("Completed and saved for all clients") 

    os.system('/opt/home/jsanchez/dev/src/push_file_v2.sh /opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Estimator/ estim_algo_output.csv brioshares QMP/Monitoring/')
    logger.info("Moved output to brioshares") 


        #output_df = output_df[['trafficday', 'requesttype', 'advertiser.customerid', 'publisher.accountid', 'GBM_OK', 'algo_type', 'multiplier', 'attribute', 'attribute_level', 'matched_searches', 'original_impressions', 'estimated_impressions', 'original_clicks', 'estimated_clicks', 'original_rank', 'estimated_rank']]
        #final = final[['trafficday', 'requesttype', 'advertiser.customerid', 'publisher.accountid', 'GBM_OK', 'algo_type', 'multiplier', 'attribute', 'attribute_level', 'matched_searches', 'original_impressions', 'estimated_impressions', 'original_clicks', 'estimated_clicks', 'original_rank', 'estimated_rank']]

        #columns = output_df.columns

        #initial_statement = """ DELETE FROM bd_reports.QMP_SH_VOL_ESTIMATOR WHERE CLIENT_CUSTOMER_ID = %s """ % (config_dict['cust_id'])
        #insert_statement =  """ INSERT INTO bd_reports.QMP_SH_VOL_ESTIMATOR
        #        UPDATED_DATETIME, TRAFFICDAY, REQUESTTYPE, CLIENT_CUSTOMER_ID, 
        #        PUBLISHERACCOUNTID, ALGO_TYPE, MULTIPLIER, ATTRIBUTE, ATTRIBUTE_LEVEL, MATCHED_SEARCHES,
        #        ORIGINAL_IMPRRESSIONS, ESTIMATED_IMPRESSIONS, ORIGINAL_CLICKS, ESTIMATED_CLICKS, ORIGINAL_RANK, ESTIMATED_RANK

        #        VALUES (SYSDATE(), STR_TO_DATE(%s, "%Y-%m-%d"), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s """

        #pushDataToDB(output_df, columns, initial_statement, insert_statement, updateBatchSize = 500)


if __name__ == "__main__":
    main(sys.argv[1:])
