from estimator import *
from GBM import GBM
import warnings
warnings.filterwarnings('ignore')

test_type = 'dynamic'

config_dict = {
'accnt_id' : 569966, 'cust_id' : 59, 'vendor_key' : 1524880, # geico
#'accnt_id' : 602805, 'cust_id' : 556, 'vendor_key' : 1524930, # liberty
'attributes' : ["agebracket"], 
#'multipliers' : [1.0, 1.2, 1.4, 1.6], 
'multipliers' : [1.0],
'date_col_name' : 'trafficday', 
'n_days' : 1,
'gbm_version' : '/opt/home/abacus/ASA/AutoRun_CTR/workspace/gbm_models/ctr_auto_gbm_output_latest_v3',
'rank_pos_mismatch_thresh' : 15.0}
config_dict['t2'] = (date.today() - timedelta(1))
config_dict['t1'] = ((date.today() - timedelta(1)) - timedelta(float(config_dict['n_days'])-1))

params_dict = {"trainingData" : "/opt/home/abacus/ASA/dev/CTR_Estimation/data",
                'queryFile' : "/opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Estimator/QMP_Estimator_Configurable.txt"}

if test_type == 'static':
    data = pd.read_csv('/opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Estimator/test_data.csv')
    data['trafficday'] = pd.to_datetime(data['trafficday'])
else:
    data, n_errors = pull_hive_data(params_dict, config_dict)
    data.to_csv('test_data.csv')

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

        df = pd.merge(result_w_bids, data_modified[["trafficday", "publisher.accountid", "publisher.accountname", "encrypted_affiliate_key", "state_code", "devicetype", "qmp_profile", "searchid","agebracket","vehicle_count","currently_insured","home_owner", "requesttype"]].drop_duplicates(), 
                    left_index=True, right_on='searchid', how='left')

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

final.to_csv('estim_algo_output.csv', index=False)

os.system('/opt/home/jsanchez/dev/src/push_file_v2.sh /opt/home/dvdries/dvdries_abacus/perforce/dev/QMP/Estimator/ estim_algo_output.csv brioshares QMP/Monitoring/')
logger.info("Moved output to brioshares") 