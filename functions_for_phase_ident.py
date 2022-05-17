import networkx as nx
import pandas as pd
import numpy as np


def interpolation_function(df, k_gaps):
    """create single timelines and make interpolation of input dataframe;
    returns interpolated dataframe and devices with not enough value points"""
    # calculation frequency of data
    freq = df.time.sort_values().groupby([df.device_id, df.type]).diff().value_counts().index[0]
    # Select a common set of time points for both signals t.
    # Make sure that the min and max values of t are in the range of both t1 and t2 to avoid extrapolations.
    x = df.groupby('device_id').time.agg([np.min, np.max])
    t1 = x.amin.max()
    t2 = x.amax.min()
    t = pd.date_range(t1, t2, freq=freq)

    # exclude devices with less 'k_gaps' measurements at any phase
    x = df[(df.time >= t1) & (df.time <= t2)].groupby('device_id').type.value_counts() < k_gaps * len(t)
    dev_errors = x[x].index.get_level_values(0).unique().values
    # drop from df devices with not enough measurement points

    # create multi-index df
    df.set_index(['device_id', 'type', 'time'], inplace=True)
    df.sort_index(inplace=True)

    #  Interpolate signals so they have the same timestamps and its amount.
    dev_all = df.index.get_level_values(0).unique()  # dev_all - df that include all devices number
    res = pd.DataFrame()
    for d in dev_all:
        res_ph = pd.DataFrame()
        for ph in df.loc[d, 'value'].index.get_level_values(0).unique():
            x = (df.loc[(d, ph), 'value'].reindex(df.loc[(d, ph), 'value'].index.union(t)).interpolate('index').reindex(
                t)).to_frame()
            x['type'] = ph
            x['device_id'] = d
            res_ph = pd.concat([res_ph, x], axis=0)
        res = pd.concat([res, res_ph], axis=0)
    res.index.name = 'time'

    res = res[~res.device_id.isin(dev_errors)].copy()
    return res, dev_errors


def create_w(cor, dev_3ph):
    """create weight (w) and location (w_ind) and pairing variation (pairing_3ph) matrix"""
    w = pd.DataFrame(0, index=dev_3ph, columns=dev_3ph)  # weights
    w_ind = pd.DataFrame(0, index=dev_3ph, columns=dev_3ph)  # pair number

    # define a pairing between 3ph devices
    pairing_3ph = [
        ['Va', 'Vb', 'Vc'],
        ['Va', 'Vc', 'Vb'],
        ['Vb', 'Va', 'Vc'],
        ['Vb', 'Vc', 'Va'],
        ['Vc', 'Va', 'Vb'],
        ['Vc', 'Vb', 'Va'],
    ]

    p = np.zeros(6)
    for i in dev_3ph:
        for j in dev_3ph:
            for n in range(0, 6):
                p[n] = cor.loc[(i, 'Va'), (j, pairing_3ph[n][0])] + \
                       cor.loc[(i, 'Vb'), (j, pairing_3ph[n][1])] + \
                       cor.loc[(i, 'Vc'), (j, pairing_3ph[n][2])]
            w.loc[i, j] = max(p)
            w_ind.loc[i, j] = p.argmax()

    return w, w_ind, pairing_3ph


def df_phases_3ph(w, dev, w_ind, pairing_3ph, df_phases):
    """create df with true phases , weight, is_changed FOR 3PH meters"""
    # Create the graph and invoke the spanning tree method
    w_i = w.loc[dev, dev]
    pearson_net = nx.from_pandas_adjacency(w_i)
    tree = nx.maximum_spanning_tree(pearson_net)

    # make tree.edges readable (convert to df)
    df_edges = pd.concat(
        [pd.DataFrame([[i[0], i[1], i[2]['weight']], ], columns=['dev_start', 'dev_end', 'weight']) for i in
         tree.edges(data=True)], ignore_index=True)

    # in loop find Va for 1st device and use that index to allocate dev2 phase according to pairing_3ph
    tmp = [(df_phases.index[0])]
    abc = df_phases.iloc[0, 0:3].values
    while not df_edges.empty:
        while df_edges.isin([tmp[0]]).values.any():
            row = df_edges[df_edges.isin([tmp[0]])].stack().index[0]  # index of first True
            ind = row[0]  # index of starting node in df_edges
            d1 = df_edges.loc[ind, row[1]]  # start device of current edge
            if row[1] == 'dev_start':
                d2 = df_edges.loc[ind, 'dev_end']  # end device of current edge
            else:
                d2 = df_edges.loc[ind, 'dev_start']

            for i in range(0, 3):
                ch = pairing_3ph[w_ind.loc[d1, d2]][i]  # first phase in pairing: 'V..'
                pos = (df_phases.loc[d1, :] == abc[i]).argmax()
                df_phases.loc[d2, abc[pos]] = ch

            df_phases.loc[d2, 'w'] = round(w.loc[d1, d2] / 3, 2)

            tmp.append(d2)
            df_edges.drop(index=ind, axis='index', inplace=True)
        del tmp[0]
    df_phases['is_changed'] = ~(df_phases.loc[:, abc] == abc).all(1)

    return df_phases


def ph1_decoder(df_phases, cor):
    """IDENTIFY PHASES FOR 1PH DEVICES BASED ON 3PH"""
    abc = df_phases.iloc[0, 0:3].values
    for ind1 in cor.columns:
        ind3 = cor.loc[:, ind1].argmax()  # max corr with 3ph device
        d1 = cor.index[ind3][0]
        ph = cor.index[ind3][1]
        pos = (df_phases.loc[d1, :] == ph).argmax()
        df_phases.loc[ind1[0], abc[pos]] = ind1[1]
        df_phases.loc[ind1[0], 'w'] = round(cor.loc[:, ind1].max(), 2)
        df_phases.loc[ind1[0], 'is_changed'] = (ind1[1] != abc[pos])

    return df_phases
