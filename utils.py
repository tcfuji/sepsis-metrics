import pandas as pd
import numpy as np
from ps_datasci import psUtilities
from ps_datasci.scripts import psClean

"""
Corey's functions
"""

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

def has_codes(codes,search_for=set(sep_codes)):
    if isinstance(codes,list) and len(codes) > 0:
        return list(set(codes).intersection(search_for))
    else:
        return []

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

"""
Ted's functions
"""

# metric function
def metric(num, denom):
    try:
        return float(num) / (num + denom)
    except ZeroDivisionError:
        return 0

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
