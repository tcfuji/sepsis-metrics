from pandas import Timedelta
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient
from ps_datasci import psUtilities
from ps_datasci.scripts import psClean
from collections import OrderedDict

import pandas as pd
import numpy as np
import time
import cPickle
import bz2
import yaml
import os



def ts_SIRS(doc,label_name='SIRS_LABEL',rm_icu=False):
    '''Systematic Inflamatory Response'''
    d = psUtilities.b2df(doc)

    # Heal the non-cleaned docs #
    if d.index.name != 'WCT':
        doc = psClean.cr((cf,doc))
        d = psUtilities.b2df(doc)

    start_at = pd.NaT #d['ADM_DATE']
    last_pos_time = pd.NaT #d['ADM_DATE']

    d[label_name] = 0
    allfalse = np.array([False for i in range(d.shape[0])])

    if 'Temperature (degrees F)' in list(d.columns):
        id_b_temp = np.logical_or(d['Temperature (degrees F)'] > 100.4, d['Temperature (degrees F)'] < 96.8)
    else:
        id_b_temp = allfalse
    if 'Respirations (breaths/min)' in list(d.columns):
        id_b_RR = d['Respirations (breaths/min)'] > 20
    else:
        id_b_RR = allfalse
    if 'Heart Rate (beats/min)' in list(d.columns):
        id_b_HR = d['Heart Rate (beats/min)'] > 90
    else:
        id_b_HR = allfalse
    if 'White Blood Cell' in list(d.columns):
        id_b_WBC = np.logical_or(d['White Blood Cell'] > 12,d['White Blood Cell'] < 4)
    else:
        id_b_WBC = allfalse

    if 'P CO2 Arterial' in list(d.columns):
        id_b_CO2 = d['P CO2 Arterial'] < 32
    else:
        id_b_CO2 = allfalse

    score = id_b_temp.astype(int) +\
        id_b_RR +\
        id_b_HR +\
        id_b_WBC +\
        id_b_CO2

    current_first_pos = d.index.max()

    id_b = score >= 2
    if id_b.sum() > 0:
        pos_times = d.index[id_b]
        first_pos_time = pos_times.min()
        last_pos_time = pos_times.max()
        if first_pos_time < current_first_pos:
            start_at = first_pos_time
            d[label_name][start_at:last_pos_time] = 1

    id_b = score >= 2
    if id_b.sum() > 0:
        d[label_name][id_b] = 1
    d[label_name + '_SCORE'] = score
    d[label_name + '_START_TIME'] = start_at
    d[label_name + '_STOP_TIME'] = last_pos_time

    #optionally strip out ICU visits
    if rm_icu:
        try:
            d[label_name][d['UNIT_MASTER_CARE_TYPE'] == 'CRITICAL CARE-ADULT'] = -1
        except:
            pass

    if (d[label_name]==1).sum() > 0:
        has_true_label=True
    else:
        has_true_label=False

    doc['data_ts'] = d

    return has_true_label

def ts_INF(doc,label_name='INFECTION_LABEL',rm_icu=False):
    '''Documented infection'''
    d = doc['data_ts']
    d[label_name] = 0
    allfalse = np.array([False for i in range(d.shape[0])])

    start_at = pd.NaT #d['ADM_DATE']
    last_pos_time = pd.NaT  #d['ADM_DATE']

    if 'Blood Culture' in list(d.columns):
        id_b_bc = d['Blood Culture'] == 1
    else:
        id_b_bc = allfalse
    if 'Urine Culture' in list(d.columns):
        id_b_uc = d['Urine Culture'] == 1
    else:
        id_b_uc = allfalse

    score = (1* id_b_bc +\
        id_b_uc)

    current_first_pos = d.index.max()

    id_b = score >= 1
    if id_b.sum() > 0:
        pos_times = d.index[id_b]
        first_pos_time = pos_times.min()
        last_pos_time = pos_times.max()
        if first_pos_time < current_first_pos:
            start_at = first_pos_time
            d[label_name][start_at:] = 1

    id_b = score >= 1
    if id_b.sum() > 0:
        d[label_name][id_b] = 1
    d[label_name + '_SCORE'] = score
    d[label_name + '_START_TIME'] = start_at
    d[label_name + '_STOP_TIME'] = last_pos_time

    #optionally strip out ICU visits
    if rm_icu:
        try:
            d[label_name][d['UNIT_MASTER_CARE_TYPE'] == 'CRITICAL CARE-ADULT'] = -1
        except:
            pass

    if (d[label_name]==1).sum() > 0:
        has_true_label=True
    else:
        has_true_label=False

    doc['data_ts'] = d
    return has_true_label

def ts_OD(doc,label_name='ORGAN_DYSFUNCTION_LABEL',rm_icu=False):
    '''Organ dysfunction'''
    d = doc['data_ts']
    d[label_name] = 0
    allfalse = np.array([False for i in range(d.shape[0])])

    start_at = pd.NaT  #d['ADM_DATE']
    last_pos_time = pd.NaT #d['ADM_DATE']

    if 'Creatinine' in list(d.columns):
        id_b_cr = d['Creatinine'] > 2
    else:
        id_b_cr = allfalse
    if 'Bilirubin Total' in list(d.columns):
        id_b_bil = d['Bilirubin Total'] > 2
    else:
        id_b_bil = allfalse
    if 'Platelet' in list(d.columns):
        id_b_pl = d['Platelet'] <= 100
    else:
        id_b_pl = allfalse

    if 'INR (INTERNATIONAL NORMALIZED RATIO)' in list(d.columns):
        id_b_INR = d['INR (INTERNATIONAL NORMALIZED RATIO)'] > 1
    else:
        id_b_INR = allfalse
    if 'Lactic Acid Level' in list(d.columns):
        id_b_LAC = d['Lactic Acid Level']>=2.2
    else:
        id_b_LAC = allfalse

    score = (1*id_b_cr +\
        id_b_bil +\
        id_b_pl +\
        id_b_INR +\
        id_b_LAC)

    current_first_pos = d.index.max()

    id_b = score >= 1
    if id_b.sum() > 0:
        pos_times = d.index[id_b]
        first_pos_time = pos_times.min()
        last_pos_time = pos_times.max()
        if first_pos_time < current_first_pos:
            start_at = first_pos_time
            d[label_name][start_at:] = 1

    id_b = score >= 1
    if id_b.sum() > 0:
        d[label_name][id_b] = 1
    d[label_name + '_SCORE'] = score
    d[label_name + '_START_TIME'] = start_at
    d[label_name + '_STOP_TIME'] = last_pos_time

    #optionally strip out ICU visits
    if rm_icu:
        try:
            d[label_name][d['UNIT_MASTER_CARE_TYPE'] == 'CRITICAL CARE-ADULT'] = -1
        except:
            pass

    if (d[label_name]==1).sum() > 0:
        has_true_label=True
    else:
        has_true_label=False

    doc['data_ts'] = d
    return has_true_label

def ts_SEP(doc,label_name='SEPSIS_LABEL',rm_icu=True):
    '''Sepsis'''
    d = doc['data_ts']
    d[label_name] = 0
    allfalse = np.array([False for i in range(d.shape[0])])

    start_at = pd.NaT #d['ADM_DATE']
    last_pos_time = pd.NaT #d['ADM_DATE']

    if 'SIRS_LABEL' in list(d.columns):
        id_b_sir = d['SIRS_LABEL']== 1
    else:
        id_b_sir = allfalse
    if 'INFECTION_LABEL' in list(d.columns):
        id_b_inf = d['INFECTION_LABEL']== 1
    else:
        id_b_inf = allfalse

    score = (1* id_b_sir +\
        id_b_inf) #+\
        #id_b_LAC)

    current_first_pos = d.index.max()

    id_b = score >= 2
    if id_b.sum() > 0:
        pos_times = d.index[id_b]
        first_pos_time = pos_times.min()
        last_pos_time = pos_times.max()
        if first_pos_time < current_first_pos:
            start_at = first_pos_time
            d[label_name][start_at:] = 1

    id_b = score >= 2
    if id_b.sum() > 0:
        d[label_name][id_b] = 1
    d[label_name + '_SCORE'] = score
    d[label_name + '_START_TIME'] = start_at
    d[label_name + '_STOP_TIME'] = last_pos_time

    #optionally strip out ICU visits
    if rm_icu:
        try:
            d[label_name][d['UNIT_MASTER_CARE_TYPE'] == 'CRITICAL CARE-ADULT'] = -1
        except:
            pass

    if (d[label_name]==1).sum() > 0:
        has_true_label=True
    else:
        has_true_label=False
    doc['data_ts'] = d
    return has_true_label

def ts_SEV(doc,label_name='SEVERE_SEPSIS_LABEL',rm_icu=False):
    '''Severe Sepsis'''
    d = doc['data_ts']
    d[label_name] = 0
    allfalse = np.array([False for i in range(d.shape[0])])

    start_at = pd.NaT #d['ADM_DATE']
    last_pos_time = pd.NaT #d['ADM_DATE']

    if 'SEPSIS_LABEL' in list(d.columns):
        id_b_sep = d['SEPSIS_LABEL']== 1
    else:
        id_b_sep = allfalse

    if 'ORGAN_DYSFUNCTION_LABEL' in list(d.columns):
        id_b_of = d['ORGAN_DYSFUNCTION_LABEL']== 1
    else:
        id_b_of = allfalse

    score = (1*id_b_sep +id_b_of)


    current_first_pos = d.index.max()

    id_b = score >= 2
    if id_b.sum() > 0:
        pos_times = d.index[id_b]
        first_pos_time = pos_times.min()
        last_pos_time = pos_times.max()
        if first_pos_time < current_first_pos:
            start_at = first_pos_time
            d[label_name][start_at:] = 1

    id_b = score >= 2
    if id_b.sum() > 0:
        d[label_name][id_b] = 1
    d[label_name + '_SCORE'] = score
    d[label_name + '_START_TIME'] = start_at
    d[label_name + '_STOP_TIME'] = last_pos_time


    if (d[label_name]==1).sum() > 0:
        has_true_label=True
    else:
        has_true_label=False
    doc['data_ts'] = d
    return has_true_label

def ts_SS(doc,label_name='SEPTIC_SHOCK_LABEL',rm_icu=False):
    '''Septic Shock'''
    d = doc['data_ts']
    d[label_name] = 0
    allfalse = np.array([False for i in range(d.shape[0])])

    start_at = pd.NaT #doc['ADM_DATE']
    last_pos_time = pd.NaT #doc['ADM_DATE']

    if 'SEVERE_SEPSIS_LABEL' in list(d.columns):
        id_b_ss = d['SEVERE_SEPSIS_LABEL']== 1
    else:
        id_b_ss = allfalse

    if 'BP Noninvasive Systolic (mm Hg)' in list(d.columns):
        id_b_BP = d['BP Noninvasive Systolic (mm Hg)'] < 90
    else:
        id_b_BP = allfalse

    score = id_b_ss + id_b_BP * 1

    current_first_pos = d.index.max()

    id_b = score >= 2
    if id_b.sum() > 0:
        pos_times = d.index[id_b]
        first_pos_time = pos_times.min()
        last_pos_time = pos_times.max()
        if first_pos_time < current_first_pos:
            start_at = first_pos_time
            d[label_name][start_at:] = 1

    d[label_name + '_SCORE'] = score
    d[label_name + '_START_TIME'] = start_at
    d[label_name + '_STOP_TIME'] = last_pos_time

    if (d[label_name]==1).sum() > 0:
        has_true_label=True
    else:
        has_true_label=False
    doc['data_ts'] = d
    return has_true_label

def time_of_CMS_severe_or_shock(doc,
                                label_name='SHOCK_OR_SEV_LABEL'):
    ts_SIRS(doc)
    ts_INF(doc)
    ts_OD(doc)
    ts_SEP(doc)
    is_sev = ts_SEV(doc)
    is_ss = ts_SS(doc)

    start_at = pd.NaT
    id_b = np.logical_or(doc['data_ts']['SEPTIC_SHOCK_LABEL'] == 1, doc['data_ts']['SEVERE_SEPSIS_LABEL'] == 1)
    if id_b.sum() > 0:
        first_pos_time = doc['data_ts'][id_b].index.min()
        start_at = first_pos_time

    return start_at

def CMS_severe_or_shock(doc,
                        label_name='SHOCK_OR_SEV_LABEL',
                        leadtime_hours=0,
                        posttime_hours=3):
    ts_SIRS(doc)
    ts_INF(doc)
    ts_OD(doc)
    ts_SEP(doc)
    is_sev = ts_SEV(doc)
    is_ss = ts_SS(doc)

    id_b = np.logical_or(doc['data_ts']['SEPTIC_SHOCK_LABEL'] == 1,doc['data_ts']['SEVERE_SEPSIS_LABEL'] == 1)

    first_pos_time = doc['data_ts'][id_b].index.min()
    start_at = first_pos_time - np.timedelta64(leadtime_hours, 'h')
    end_at = first_pos_time + np.timedelta64(posttime_hours, 'h')


    if id_b.sum() > 0:
        doc['data_ts'][label_name] = -2
    else:
        doc['data_ts'][label_name] = 0

    doc['data_ts'][label_name][start_at:end_at] = 1
    min_ind = doc['data_ts'][label_name][start_at:end_at].index.min()

    # Exclude POA
    if first_pos_time == doc['data_ts'].index.min():
        doc['data_ts'][label_name] = -2

    icu = doc['data_ts']['SEPSIS_LABEL'] == -1
    doc['data_ts'][label_name][icu] = -1

    has_pos = (doc['data_ts'][label_name] == 1).sum() > 0
    doc['data_ts'] = psUtilities.df2b(doc['data_ts'])

    return has_pos, min_ind

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

def has_codes(codes,search_for=set(sep_codes)):
    if isinstance(codes,list) and len(codes) > 0:
        return list(set(codes).intersection(search_for))
    else:
        return []

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

# metric function

def metric(num, denom):
    try:
        return float(num) / (num + denom)
    except ZeroDivisionError:
        return 0

"""
# metrics
def sensitivity(tp, fn):
    return float(tp) / (tp + fn)

def specificity(tn, fp):
    return float(tn) / (tn + fp)

def ppv(tp, fp):
    return float(tp) / (tp + fp)

def npv(tn, fn):
    return float(tn) / (tn + fn)
"""
# Return a list of tuples each containing a time interval.
def time_interval_list(start_date, end_date, time_interval):
    interval_list = []
    time_index = []
    current_date = start_date
    while current_date <= end_date:
        time_index.append(current_date)
        interval_list.append((current_date - time_interval, current_date))
        current_date += time_interval
    return time_index, interval_list

class TimeIntervalDict:
    def __init__(self, time_intervals):
        self.time_dict = dict()
        for t in time_intervals:
            self.time_dict[t] = []
        self.time_dict = OrderedDict(sorted(self.time_dict.items(), key=lambda t: t[0]))

    def insert(self, time, item):
        time_not_included = True
        for t1, t2 in self.time_dict:
            if t1 <= time < t2:
                self.time_dict[(t1, t2)].append(item)
                time_not_included = False
                # self.time_dict = OrderedDict(sorted(self.time_dict.items(), key=lambda t: t[0]))
                break
        if time_not_included:
            print(str(time) + " is not in any of the intervals!")

    def get(self, time):
        for t1, t2 in self.time_dict:
            if t1 <= time < t2:
                return self.time_dict[(t1, t2)]
        print(str(time) + " is not in any of the intervals!")

    def keys(self):
        return [key for key in self.time_dict.keys()]

    def values(self):
        return [val for val in self.time_dict.values()]

def mk_dir(s):
    try:
        os.mkdir(s)
    except OSError:
        pass

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
