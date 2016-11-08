# coding=utf-8
from collections import OrderedDict
import pandas as pd
import pymc3 as pm
from simpanel.glm import Glm


class SimPanel(object):
    def __init__(self, name, data, target, sim=None, idindex=None, idcol=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Please provide X as pd.DataFrame and y as pd.Series')
        if idcol is None and idindex is None:
            raise ValueError('Please provide idcol or idindex')
        self.name = name
        self.ylabel = target
        self.groups = OrderedDict()
        self.advifits = OrderedDict()
        level = idindex
        if idcol:
            data = data.set_index(idcol, append=True)
            level = idcol
        for label, df in data.groupby(level=level):
            self.groups[label] = df
        self.sim = sim

    @property
    def data(self):
        return pd.concat(self.groups.values())
