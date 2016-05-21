from pandas import Timedelta
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient
from collections import OrderedDict

import pandas as pd
import numpy as np
import time
import cPickle
import bz2
import yaml
import os

from utils import *

deceased_or_hospice =['EXPIRED',
                      'D/C INP HOSPICE FAC',
                      'D/C HOSPICE',
                      'EXPIRED MED FAC',
                      'D/C ACUTE CARE HOSP',
                      'D/C SHORT TERM HOSP',
                      'D/C VA/FED HOSP']

sep_codes = ['A41.9',
 'R65.10',
 'R65.11',
 'R65.20',
 'R65.21',
 'R78.81',
 '995.90',
 '995.91',
 '995.92',
 '785.52',
 '999.91',
 '790.7',
 '995.94',
 '995.93']

def write_metrics(start_date, end_date, time_interval):
    import time

    # initialize lists for the csv file.
    tp_dh_list = []
    tn_dh_list = []
    fp_dh_list = []
    fn_dh_list = []

    sens_dh_list = []
    spec_dh_list = []
    ppv_dh_list = []
    npv_dh_list = []

    # string_vns = []

    # Get the collections
    preds = client.psPreds.preds
    encounters = client.psRawTS_inp.encounters

    time_index, intervals = time_interval_list(start_date, end_date, time_interval)

    tp_time_dict = TimeIntervalDict(intervals)
    fp_time_dict = TimeIntervalDict(intervals)
    tn_time_dict = TimeIntervalDict(intervals)
    fn_time_dict = TimeIntervalDict(intervals)
    beginning_time = intervals[0][0]

    all_vns = set()

    # For each time interval...
    for start_time, end_time in tqdm(intervals):
        s = time.time()

        mk_dir('data')
        mk_dir('data/chunks')
        mk_dir('data/ts')

        print("Finding all patients discharged within " + str(start_time) + " to " + str(end_time))

        # Finds all visit numbers of patients discharged within the current time interval.
        q = {'DISCHARGE_DATE': {'$gte': start_time, '$lt': end_time}}
        p = {'DISCHARGE_DATE': 1, 'DISCHARGE_STATUS': 1, 'VISIT_NUMBER':1, 'data_ts': 1, '_id':0}
        current_discharged_patients = encounters.find(q, p)
        current_discharged_patients = [d for d in current_discharged_patients if type(d['VISIT_NUMBER']) == int]
        current_discharged_vns = [d['VISIT_NUMBER'] for d in current_discharged_patients]


        print("Starting query on preds to find patients who were alerted.")
        # 'WCT': {'gte': beginning_time, '$lt': end_time}
        alert_q = {'WCT': {'$gte': beginning_time, '$lt': end_time}, 'VISIT_NUMBER':{'$in':current_discharged_vns}, \
                'modelName':'sepsismodel', 'alertPush': True}
        alerted = preds.find(alert_q, {'WCT': 1, 'VISIT_NUMBER': 1})
        alerted = [d for d in alerted]

        print("Chunking the non-alerted preds")
        # Now more fault-tolerant
        not_alerted = []
        i = 0
        for vn in chunks(current_discharged_vns, 10):
            try:
                with bz2.BZ2File('data/chunks/{}_{}_{}_test.pkl.bz2'.format(str(start_time), time_interval, i), 'rb') as f:
                    docs_list = cPickle.load(f)
                    # print(docs_list)
                    not_alerted.extend(docs_list)
            except (IOError, EOFError):
                not_alerted_q = {'WCT': {'$gte': beginning_time, '$lt': end_time}, 'VISIT_NUMBER':{'$in':vn}, \
                        'modelName':'sepsismodel', 'alertPush': False}
                cursor = preds.find(not_alerted_q, {'WCT': 1, 'VISIT_NUMBER': 1})
                docs_list = [d for d in cursor]
                with bz2.BZ2File('data/chunks/{}_{}_{}_test.pkl.bz2'.format(str(start_time), time_interval, i), 'wb') as f:
                    cPickle.dump(docs_list, f, -1)
                not_alerted.extend(docs_list)
            i += 1

        print("Converting to sets.")
        alerted_vns = set([d['VISIT_NUMBER'] for d in alerted if type(d['VISIT_NUMBER']) == int])
        not_alerted_vns = set([d['VISIT_NUMBER'] for d in not_alerted if type(d['VISIT_NUMBER']) == int])

        # Find time series and split docs to either sepsis or no sepsis.
        # Compare predictions with sepsis time series and sort by dates.
        # Remember to omit sepsis signals before beginning_time.
        # Also splits into tp, tn, fp, fn.
        enc_alert_docs = []
        enc_no_alert_docs = []
        time_ind_dict = {}
        try:
            with bz2.BZ2File('data/ts/{}_{}_time_ind_dict.pkl.bz2'.format(str(start_time), time_interval), 'rb') as f:
                time_ind_dict = cPickle.load(f)
        except (IOError, EOFError):
            pass

        print("Splitting time series")
        for doc in tqdm(current_discharged_patients):
            vn = doc['VISIT_NUMBER']
            if vn in all_vns:
                continue
            try:
                time_ind = time_ind_dict[vn]
            except KeyError:
                time_ind = time_of_CMS_severe_or_shock(doc)
                time_ind_dict[vn] = time_ind
            # Ignores patient who developed sepsis before beginning_time
            if time_ind < beginning_time:
                continue
            if pd.notnull(time_ind):
                print("Has sepsis")
                if vn in alerted_vns:
                    # The patient had sepsis and was alerted.
                    print('tp')
                    tp_time_dict.insert(time_ind, doc)
                    enc_alert_docs.append(doc)
                elif vn in not_alerted_vns:
                    # The patient had sepsis but was not alerted this week.
                    # OK since we omit patients who were discharged AND
                    # had sepsis before beginning_time.
                    print("fn")
                    fn_time_dict.insert(time_ind, doc)
                    enc_no_alert_docs.append(doc)
            else:
                print("Does not have sepsis")
                if vn in alerted_vns:
                    # Was alerted but didn't contract sepsis
                    print("fp")
                    fp_time_dict.insert(start_time, doc)
                    enc_alert_docs.append(doc)
                elif vn in not_alerted_vns:
                    # Didn't have sepsis and was never alerted after beginning_time.
                    print("tn")
                    tn_time_dict.insert(start_time, doc)
                    enc_no_alert_docs.append(doc)
            all_vns.add(vn)
        with bz2.BZ2File('data/ts/{}_{}_time_ind_dict.pkl.bz2'.format(str(start_time), time_interval), 'wb') as f:
            cPickle.dump(time_ind_dict, f, -1)




        enc_alert_df = pd.DataFrame(enc_alert_docs)
        enc_no_alert_df = pd.DataFrame(enc_no_alert_docs)

        # Add dataframe columns indicating if the patients were correctly
        # diagnosed or classified as deceased or hospice.
        print("Looking for this month's deceased or hospice patients who were under the sepsis model")
        try:
            enc_alert_df['deceased_or_hospice'] = \
                enc_alert_df.DISCHARGE_STATUS.isin(deceased_or_hospice)
        except AttributeError:
            enc_alert_df['deceased_or_hospice'] = pd.Series()
        try:
            enc_no_alert_df['deceased_or_hospice'] = \
                enc_no_alert_df.DISCHARGE_STATUS.isin(deceased_or_hospice)
        except AttributeError:
            # In case DISCHARGE_STATUS does not exist.
            enc_no_alert_df['deceased_or_hospice'] = pd.Series()

        total_alerts = enc_alert_df.shape[0]
        total_no_alerts = enc_no_alert_df.shape[0]

        print("There were " + str(total_alerts) + " dh patients who received alerts this interval")
        print("There were " + str(total_no_alerts) + " dh patients who did not receive alerts this interval")

        # Calculate the outcomes and metrics
        # TP: patients who received an alert and had DH code
        true_positives_dh = enc_alert_df.deceased_or_hospice.sum()
        # FP: patients who received an alert and had no DH code
        false_positives_dh = total_alerts - true_positives_dh
        # FN: patients who did not receive an alert and had DH code
        false_negatives_dh = enc_no_alert_df.deceased_or_hospice.sum()
        # TN: patients who did not receive an alert and had no DH code
        true_negatives_dh = total_no_alerts - false_negatives_dh

        print("The number of deceased or hospice true positives is: ")
        print(str(true_positives_dh) + '\n')
        print("The number of deceased or hospice false positives is: ")
        print(str(false_positives_dh) + '\n')
        print("The number of deceased or hospice true negatives is: ")
        print(str(true_negatives_dh) + '\n')
        print("The number of deceased or hospice false negatives is: ")
        print(str(false_negatives_dh) + '\n')

        sens_dh = metric(true_positives_dh, false_negatives_dh)
        spec_dh = metric(true_negatives_dh, false_positives_dh)
        ppv_dh = metric(true_positives_dh, false_positives_dh)
        npv_dh = metric(true_negatives_dh, false_negatives_dh)

        print("The sensitivity of deceased or hospice is: ")
        print(str(sens_dh) + '\n')
        print("The specificity of deceased or hospice is: ")
        print(str(spec_dh) + '\n')
        print("The positive predictive value of deceased or hospice is: ")
        print(str(ppv_dh) + '\n')
        print("The negative predictive value of deceased or hospice is: ")
        print(str(npv_dh) + '\n')


        # Add values and metrics to lists
        tp_dh_list.append(true_positives_dh)
        tn_dh_list.append(true_negatives_dh)
        fp_dh_list.append(false_positives_dh)
        fn_dh_list.append(false_negatives_dh)

        sens_dh_list.append(sens_dh)
        spec_dh_list.append(spec_dh)
        ppv_dh_list.append(ppv_dh)
        npv_dh_list.append(npv_dh)

        print("%f seconds" % (time.time() - s))

    # Calculate metrics for time series.
    tp_sep_list = []
    tn_sep_list = []
    fp_sep_list = []
    fn_sep_list = []

    sens_sep_list = []
    spec_sep_list = []
    ppv_sep_list = []
    npv_sep_list = []

    print("Sorting out time series metrics")
    for time, _ in tp_time_dict.keys():
        tp = len(tp_time_dict.get(time))
        fn = len(fn_time_dict.get(time))
        tn = len(tn_time_dict.get(time))
        fp = len(fp_time_dict.get(time))

        tp_sep_list.append(tp)
        tn_sep_list.append(tn)
        fp_sep_list.append(fp)
        fn_sep_list.append(fn)

        sens = metric(tp, fn)
        spec = metric(tn, fp)
        ppv = metric(tp, fp)
        npv = metric(tn, fn)

        sens_sep_list.append(sens)
        spec_sep_list.append(spec)
        ppv_sep_list.append(ppv)
        npv_sep_list.append(npv)

        print("The number of deceased or hospice true positives is: ")
        print(str(tp) + '\n')
        print("The number of deceased or hospice false positives is: ")
        print(str(fp) + '\n')
        print("The number of deceased or hospice true negatives is: ")
        print(str(tn) + '\n')
        print("The number of deceased or hospice false negatives is: ")
        print(str(fn) + '\n')

        print("The sensitivity of deceased or hospice is: ")
        print(str(sens) + '\n')
        print("The specificity of deceased or hospice is: ")
        print(str(spec) + '\n')
        print("The positive predictive value of deceased or hospice is: ")
        print(str(ppv) + '\n')
        print("The negative predictive value of deceased or hospice is: ")
        print(str(npv) + '\n')

    # Create dataframes for easy writing to csv's
    dh_df = pd.DataFrame({'true_positives': tp_dh_list, 'true_negatives': tn_dh_list,
            'false_positives': fp_dh_list, 'false_negatives': fn_dh_list}, index=time_index)
    sep_df = pd.DataFrame({'true_positives': tp_sep_list, 'true_negatives': tn_sep_list,
            'false_positives': fp_sep_list, 'false_negatives': fn_sep_list}, index=time_index)
    dh_metric_df = pd.DataFrame({'sensitivity': sens_dh_list, 'specificity': spec_dh_list,
            'ppv': ppv_dh_list, 'npv': npv_dh_list}, index=time_index)
    sep_metric_df = pd.DataFrame({'sensitivity': sens_sep_list, 'specificity': spec_sep_list,
            'ppv': ppv_sep_list, 'npv': npv_sep_list}, index=time_index)

    # write to csv's
    try:
        os.mkdir('results')
    except OSError:
        pass
    dh_df.to_csv('results/deceased_or_hospice_outcomes.csv')
    sep_df.to_csv('results/has_sepsis_outcomes.csv')
    dh_metric_df.to_csv('results/deceased_or_hospice_metrics.csv')
    sep_metric_df.to_csv('results/has_sepsis_metrics.csv')

    # print("There are " + str(len(string_vns)) + " string vns.")

if __name__ == '__main__':
    # Make sure you're connected to the database
    with open('mongo_creds.yaml') as f:
        creds = yaml.safe_load(f)
    client = MongoClient('UPHSVLNDC058.uphs.upenn.edu', port=27017)
    is_authed = client.admin.authenticate(creds['user'],creds['pass'])

    assert is_authed
    cf = psUtilities.load_config()
    """
    1. User will be asked to enter start and end dates. These dates will be represented
    as datetimes.
    2. User will be asked to enter desired time interval. This interval will be
    represented as a pandas.Timedelta data structure.
    """

    # Comment out until testing is done.
    while True:
        print("Specify starting date (e.g. 2016/12/31): ")
        start_date = raw_input("Enter here: ").strip().split('/')
        start_date = datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]))

        print("Specify end date (e.g. 2016/12/31): ")
        end_date = raw_input("Enter here: ").strip().split('/')
        end_date = datetime(int(end_date[0]), int(end_date[1]), int(end_date[2]))

        if start_date < end_date:
            break
        else:
            print("Invalid ordering of dates!!")
            print("Make sure the starting date comes before the end date! \n")


    while True:
        print("What is your desired interval (in terms of days, hours or seconds)?")
        print("E.g. '30 days', '7 days', '1 day', '1 day 2 hours'")

        try:
            time_interval = raw_input("Enter here: ")
            time_interval = Timedelta(time_interval)
            break
        except ValueError:
            print("Please enter a valid time interval.")

    overall_time = time.time()
    # write_metrics(datetime(2015, 10, 1), datetime(2015, 11, 1), Timedelta('7 days'))
    write_metrics(start_date, end_date, time_interval)
    print("%f seconds" % (time.time() - overall_time))
