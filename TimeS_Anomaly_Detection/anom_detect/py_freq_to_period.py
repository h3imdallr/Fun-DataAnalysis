from statsmodels.compat.python import range, lrange, lzip
import numpy as np
import numpy.lib.recfunctions as nprf
from statsmodels.tools.tools import add_constant
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def freq_to_period(freq):
    """
    Convert a pandas frequency to a periodicity

    Parameters
    ----------
    freq : str or offset
        Frequency to convert

    Returns
    -------
    period : int
        Periodicity of freq

    Notes
    -----
    Annual maps to 1, quarterly maps to 4, monthly to 12, weekly to 52.
    """
    if not isinstance(freq, offsets.DateOffset):
        freq = to_offset(freq)  # go ahead and standardize
    freq = freq.rule_code.upper()

    if freq == 'A' or freq.startswith(('A-', 'AS-')):
        return 1
    elif freq == 'Q' or freq.startswith(('Q-', 'QS-')):
        return 4
    elif freq == 'M' or freq.startswith(('M-', 'MS')):
        return 12
    elif freq == 'W' or freq.startswith('W-'):
        return 52

    #ymjung
    elif freq =='D':
        return 7
    elif freq == 'H':
        # return 8760
        return 24
    elif freq == 'T':
        # return 525600
        return 1440

    else:  # pragma : no cover
        raise ValueError("freq {} not understood. Please report if you "
                         "think this in error.".format(freq))