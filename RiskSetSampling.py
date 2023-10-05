from datetime import datetime

import numpy as np
import pandas as pd

def convert_date_to_int(date_string, format = "%Y-%m-%d"):
    date_object = datetime.strptime(date_string, format)
    return date_object.timestamp()
def RiskSetSampling(
        data,
        entry,
        exit,
        fail,
        origin = '',
        controls = 1,
        match = '',
        include = list(),
        silent = False,
        r_seed=0):
    '''
    Generate a nested case-control study:
    Given the basic outcome variables for a cohort study: the time of
    entry to the cohort, the time of exit and the reason for exit
    ("failure" or "censoring"), this function computes risk sets and
    generates a matched case-control study in which each case is
    compared with a set of controls randomly sampled from the
    appropriate risk set. Other variables may be matched when
    selecting controls.
    :param entry:
    :param exit:
    :param fail:
    :param origin:
    :param controls:
    :param match:
    :param include:
    :param data:
    :param silent:
    :param r_seed:
    :return:
    '''
    np.random.seed(r_seed)
    entry = data[entry].apply(convert_date_to_int)
    exit = data[exit].apply(convert_date_to_int)
    fail = data[fail]
    origin = data[origin].apply(convert_date_to_int)
    n = len(fail)
    if len(exit) != n:
        raise ValueError("All vectors must have same length")
    if len(entry) != 1 and len(entry) != n:
        raise ValueError("All vectors must have same length")
    if len(origin) == 1:
        origin = [origin] * n
    else:
        if len(origin) != n:
            raise ValueError("All vectors must have same length")
    t_entry = entry - origin
    t_exit = exit - origin
    marg = match
    if type(marg) == str:
        match = {match: data[marg].tolist()}
    elif isinstance(marg, list):
        match = {m: data[m].to_list() for m in marg}
    else:
        raise ValueError("illegal argument (match)")
    m = len(match)
    mnames = list(match.keys())
    if m > 0:
        for i in range(m):
            if len(match[mnames[i]]) != n:
                raise ValueError("incorrect length for matching variable")
    iarg = include
    if isinstance(iarg, str):
        include = {include: data[include].tolist()}
    elif isinstance(iarg, list):
        include = {i: data[i].to_list() for i in iarg}
    ni = len(include)
    inames = list(include.keys())
    if ni > 0:
        for i in inames:
            if len(include[i]) != n:
                raise ValueError("incorrect length for included variable")
    if m > 0:
        grp = pd.Series(['-'.join([str(x) for x in row]) for row in
                         pd.DataFrame(match).values.tolist()])
    nn = (1 + controls) * sum(fail != 0)
    pr = pd.Series([0] * nn)
    sr = pd.Series([0] * nn)
    tr = pd.Series([0] * nn)
    fr = pd.Series([0] * nn)
    nn = 0
    if not silent:
        print("\nSampling risk sets: ")
    sets = 0
    nomatch = 0
    incomplete = 0
    ties = False
    fg = list(set(grp[i] for i in range(n) if fail[i] != 0))
    for g in fg:
        ft = list(set(t_exit[(grp == g) & (fail != 0)]))
        for tf in ft:
            if not silent:
                print(".")
            sets += 1
            case = (grp == g) & (t_exit == tf) & (fail != 0)
            ncase = sum(case)
            if ncase > 1:
                ties = True
            noncase = (grp == g) & (t_entry <= tf) & (t_exit >= tf) & ~case
            ncont = controls * ncase
            if ncont > sum(noncase):
                ncont = sum(noncase)
                if ncont > 0:
                    incomplete += 1
            if ncont > 0:
                newnn = nn + ncase + ncont
                sr[nn : newnn] = sets
                tr[nn : newnn] = tf
                fr[nn : nn + ncase] = 1
                fr[nn + ncase : newnn] = 0
                pr[nn : nn + ncase] = pd.Series(list(range(n)))[case]
                noncase.id = pd.Series(list(range(n)))[noncase]
                pr[nn + ncase : newnn] = np.random.choice(noncase.id,
                                                          size=ncont,
                                                          replace=False)
                nn = newnn
            else:
                nomatch += ncase
    if not silent:
        print("\n")
    res = [None] * (4 + m + ni)
    if nn > 0:
        res[0] = sr[0:nn].reset_index(drop=True).astype(int)
        res[1] = map = pr[0:nn].reset_index(drop=True).astype(int)
        res[2] = (tr[0:nn] + origin[map].reset_index(drop=True)).apply(
            lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d")
        )
        res[3] = fr[0:nn].reset_index(drop=True).astype(int)
    if m > 0:
        for i in range(m):
            res[4 + i] = pd.Series(match[mnames[i]])[map].reset_index(drop=True)
    if ni > 0:
        for i in range(ni):
            res[4 + m + i] = pd.Series(include[inames[i]])[map].reset_index(
                drop=True)
    res = pd.DataFrame(res).transpose()
    res.columns = ["Set", "Map", "Time", "Fail"] + mnames + inames
    res = res.convert_dtypes()
    if incomplete > 0:
        print(str(incomplete) + " case-control sets are incomplete")
    if nomatch > 0:
        print(str(nomatch) + " cases could not be matched")
    if ties:
        print("there were tied failure times")

    return pd.DataFrame(res)
