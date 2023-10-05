from datetime import datetime
import numpy as np
import pandas as pd

def convert_date_to_int(date_string, format = "%Y-%m-%d"):
    '''
    Convert date string to integer value
    :param date_string: string of date
    :param format: format of date
    :return: integer value of date
    '''
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
    The function takes into account the basic outcome variables of a cohort
    study such as the entry time, exit time, and the reason for exit (either
    "failure" or "censoring"). It then computes the risk sets and creates a
    matched case-control study where each case is compared to a set of
    controls that are randomly sampled from the appropriate risk set. You
    can also match other variables when selecting controls.
    :param data: input data frame
    :param entry: column name of entry time
    :param exit: column name of exit time
    :param fail: column name of failure indicator
    :param origin: column name of origin time
    :param controls: number of controls per case
    :param match: column name(s) of matching variable(s)
    :param include: column name(s) of additional variable(s) to include in
    the output dataframe
    :param silent: if True, suppresses output
    :param r_seed: random seed used in selection of controls
    :return:
    '''
    # set random seed for reproducibility
    np.random.seed(r_seed)
    # convert dates to integers
    entry = data[entry].apply(convert_date_to_int)
    exit = data[exit].apply(convert_date_to_int)
    fail = data[fail]
    origin = data[origin].apply(convert_date_to_int)

    # check that all vectors are the same length
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

    # compute time variables
    t_entry = entry - origin
    t_exit = exit - origin

    # check matching variables
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

    # check included variables
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
        match_lst = pd.DataFrame(match).values.tolist()
        grp = pd.Series(['-'.join([str(x) for x in row]) for row in match_lst])

    # compute risk sets
    nn = (1 + controls) * sum(fail != 0)
    pr = pd.Series([0] * nn)
    sr = pd.Series([0] * nn)
    tr = pd.Series([0] * nn)
    fr = pd.Series([0] * nn)
    nn = 0
    if not silent:
        print("\nSampling risk sets: ")

    # loop over groups
    sets = 0
    nomatch = 0
    incomplete = 0
    ties = False
    fg = list(set(grp[i] for i in range(n) if fail[i] != 0))
    fg.sort()
    for g in fg:
        # loop over failure times
        ft = list(set(t_exit[(grp == g) & (fail != 0)]))
        ft.sort()
        for tf in ft:
            # loop over cases
            if not silent:
                print(".")
            sets += 1
            # select cases in risk set (if possible)
            case = (grp == g) & (t_exit == tf) & (fail != 0)
            ncase = sum(case)
            if ncase > 1:
                ties = True
            # select controls for each case in risk set (if possible)
            noncase = (grp == g) & (t_entry <= tf) & (t_exit >= tf) & ~case
            ncont = controls * ncase

            # if not enough controls, select all available controls and add
            # to incomplete sets counter (if possible)
            if ncont > noncase.sum():
                ncont = noncase.sum()
                if ncont > 0:
                    incomplete += 1
            # if enough controls, select random sample of controls (if
            # possible)
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
                # if not enough controls, add to nomatch sets counter
                nomatch += ncase
    if not silent:
        print("\n")

    # sort clusters by risk set, time, and failure indicator
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

    # print warnings
    if incomplete > 0:
        print(str(incomplete) + " case-control sets are incomplete")
    if nomatch > 0:
        print(str(nomatch) + " cases could not be matched")
    if ties:
        print("there were tied failure times")

    return pd.DataFrame(res)
