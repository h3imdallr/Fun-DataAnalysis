# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm

# import freq_to_period_modified
import ym_seasonal


def stl(inDF, ns, np):

    res = ym_seasonal.seasonal_decompose(inDF)

    # Each components can be accssed with:
    residual = res.resid
    seasonal = res.seasonal
    trend = res.trend

    # decomposed DF
    outDF = pd.concat([residual, seasonal, trend], axis=1, keys = ['residual','seasonal','trend'])
    return outDF

