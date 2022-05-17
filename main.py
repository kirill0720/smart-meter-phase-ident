import pandas as pd
import numpy as np
from functions_for_phase_ident import interpolation_function, create_w, df_phases_3ph, ph1_decoder


class ErrMsg:
    gaps = 'Недостаточно данных измерений'
    nocorr = 'Другой источник питания'
    wkcorr = 'Слабая корреляция с группой счетчиков'
    zero = 'Нули вместо значений напряжения'
    novoltage = 'Отсутствуют данные по напряжению'


def check_phases_with_reference(df, dev_bas):
    """ This algorithm identify real phases of smart meters' voltage channels. It uses reference meter to assosiate all
    other phases.
    Based on paper: "OLIVIER, ERNST. AUTOMATIC PHASE IDENTIFICATION OF SMART METER MEASUREMENT DATA. 2017"

    INPUT
    -DataFrame with 'time', 'device_id', 'type', 'value' columns (type: ['Va', 'Vb', 'Vc'])
    dev_bas - reference device
    OUTPUT
    'df_phases' - DataFrame with phases identified
    'df_errors' - DataFrame with device_id and comment why phases are not detected
    """

    # algorithm settings [0, 1]
    k_no = 0.4  # if corr<k_no exclude device from analysis as from different balance group
    k_wk = 0.55  # if k_no<corr<k_wk exclude device from analysis as having weak correlation
    k_gaps = 0.5  # if len(data points)/total<k_gaps then exclude device from analysis

    # initialize some variables
    errmsg = ErrMsg
    errdev = {}
    devices = df.device_id.unique()
    n_total = len(devices)
    df_phases = pd.DataFrame(columns=['device_id', 'Va', 'Vb', 'Vc', 'w', 'is_changed'])
    stats = {'total': n_total,
             'analyzed': 0,
             'dev_right': 0,
             'dev_wrong': 0,
             'not_analyzed': n_total}

    df['time'] = pd.to_datetime(df['time'])

    # data cleaning: not voltage data
    df = df[df.type.isin(['Va', 'Vb', 'Vc'])]  # data contains only voltage data type
    dev_no_v = list(set(devices).difference(set(df.device_id.unique())))
    errdev.update(dict(zip(dev_no_v, [errmsg.novoltage] * len(dev_no_v))))  # update dict with error devices

    # data cleaning: exclude voltage 0-values
    dev_v_zero = df[df.value == 0].device_id.unique()
    df = df[df.value != 0]  # data contains non-zero voltage data only
    errdev.update(dict(zip(dev_v_zero, [errmsg.zero] * len(dev_v_zero))))  # update dict with error devices

    if df.empty:
        return df_phases, errdev, stats

    # create single timelines and make interpolation
    res, dev_errors = interpolation_function(df.copy(), k_gaps)

    errdev.update(dict(zip(dev_errors, [errmsg.gaps] * len(dev_errors))))  # update dict with error devices

    df = df[~df.device_id.isin(dev_errors)]  # drop from df devices with not enough measurement points

    # create 3ph- and 1ph-devices list (important that after k_gaps check)
    x = df.groupby('device_id').type.nunique()
    dev_3ph = x[x == 3].index.values
    dev_1ph = x[x == 1].index.values

    # create correlation matrix
    cor = res.set_index(['device_id', 'type'], append=True).unstack(level=[1, 2]).value.corr()

    w, w_ind, pairing_3ph = create_w(cor, dev_3ph)

    np.fill_diagonal(w.values, 0)
    # detect no correlation 3ph-devices
    x = (w.max() < k_no * 3)
    dev_no_cor = x[x].index.values
    errdev.update(dict(zip(dev_no_cor, [errmsg.nocorr] * len(dev_no_cor))))  # update dict with error devices

    # detect weak correlation 3ph-devices
    x = ((w.max() >= k_no * 3) & (w.max() < k_wk * 3))
    dev_weak_cor = x[x].index.values
    errdev.update(dict(zip(dev_weak_cor, [errmsg.wkcorr] * len(dev_weak_cor))))  # update dict with error devices

    # exclude no and weak correlation 3ph-devices from analysis
    rmv = np.concatenate((dev_no_cor, dev_weak_cor))
    w.drop(index=rmv, columns=rmv, inplace=True)
    w_ind.drop(index=rmv, columns=rmv, inplace=True)
    cor.drop(index=rmv, columns=rmv, inplace=True)
    dev_3ph = np.setdiff1d(dev_3ph, rmv)

    # initialize resulting phase-table with reference device data
    df_phases.set_index('device_id', inplace=True)
    df_phases.loc[dev_bas, :] = ['Va', 'Vb', 'Vc', 1, False]

    # %% create df with true phases , weight, is changed FOR 3PH (df_phases)
    df_phases = df_phases_3ph(w, dev_3ph, w_ind, pairing_3ph, df_phases)

    # IDENTIFY PHASES FOR 1PH DEVICES BASED ON 3PH
    cor.drop(index=dev_1ph, columns=dev_3ph, inplace=True)

    # detect no correlation 1ph-devices
    x = (cor.max(axis=0) < k_no)
    dev_no = x[x].index.get_level_values(0).values
    errdev.update(dict(zip(dev_no, [errmsg.nocorr] * len(dev_no))))  # update dict with error devices

    # detect weak correlation 1ph-devices
    x = ((cor.max(axis=0) >= k_no) & (cor.max(axis=0) < k_wk))
    dev_weak = x[x].index.get_level_values(0).values
    errdev.update(dict(zip(dev_weak, [errmsg.wkcorr] * len(dev_weak))))  # update dict with error devices

    # exclude no and weak correlation devices
    cor.drop(columns=np.concatenate((dev_no, dev_weak)), inplace=True)

    # %% IDENTIFY PHASES FOR 1PH DEVICES BASED ON 3PH (df_phases)
    df_phases = ph1_decoder(df_phases, cor)

    df_phases = df_phases.reset_index()

    # update analysis stats
    stats['analyzed'] = len(df_phases)
    stats['dev_right'] = len(df_phases) - df_phases.is_changed.sum()
    stats['dev_wrong'] = df_phases.is_changed.sum()
    stats['not_analyzed'] = n_total - len(df_phases)

    return df_phases, errdev, stats


if __name__ == "__main__":
    # prepare data
    ref_device_id = 4087261
    raw_data = pd.read_excel(r'initial_data.xlsx', header=0)

    # run calculations
    phases_tbl, errors, stat = check_phases_with_reference(raw_data, ref_device_id)
    print(phases_tbl)
    print(errors)
    print(stat)
