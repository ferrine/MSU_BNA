# coding=utf-8

import unittest

import os
import pandas as pd
import pymc3 as pm
from simpanel.glm import Glm
from simpanel.panel import SimPanel


ROOT = os.path.dirname(__file__)


class LoadData(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(os.path.join(ROOT, 'data', 'testdata.csv'))
        self.y, self.X = data.iloc[:, 0], data.iloc[:, 1:]
        self.mdata = data.iloc[:10]
        self.data = data


class TestGlm(LoadData):
    def test_init(self):
        import patsy
        formula = patsy.ModelDesc(
            [patsy.Term([patsy.LookupFactor(self.y.name)])],
            [patsy.Term([patsy.LookupFactor(p)])
             for p in self.X.columns if p != 'Country']
        )
        with pm.Model():
            _ = Glm('glm', formula, self.mdata)

    def test_init_from_xy(self):
        import numpy as np
        with pm.Model():
            _ = Glm.from_xy('glm', self.X.iloc[:, 1:-1], self.y)
        with pm.Model():
            _ = Glm.from_xy('glm', np.asarray(self.X.iloc[:, 1:-1]), self.y)
        with pm.Model():
            _ = Glm.from_xy('glm', self.X.iloc[:, 1:-1], np.asarray(self.y))
        with pm.Model():
            _ = Glm.from_xy('glm', np.asarray(self.X.iloc[:, 1:-1]), np.asarray(self.y))

    def test_advi(self):
        with pm.Model():
            aus = self.data.ix[self.data.Country == 'Australia', :-1]
            g = Glm.from_xy('glm', aus.iloc[:, 1:], aus.iloc[:, 0])
            fit = g.advi(verbose=False)
        self.assertTrue(all(fit.means.values()))

    def test_nuts(self):
        with pm.Model():
            aus = self.data.ix[self.data.Country == 'Australia', :-1]
            g = Glm.from_xy('glm', aus.iloc[:, 1:], aus.iloc[:, 0])
            trace = g.nuts(draws=10)
        self.assertTrue(all(trace))

    def test_name_does_not_overlaps(self):
        with pm.Model():
            _ = Glm.from_xy('glm1', self.X.iloc[:, 1:-1], self.y)
            _ = Glm.from_xy('glm2', self.X.iloc[:, 1:-1], self.y)


class TestPanel(LoadData):
    def test_init(self):
        _ = SimPanel('panel1', self.data,
                     'SOI_1_Ygr', idcol='Country')
        _ = SimPanel('panel2', self.data.set_index('Country', append=True),
                     'SOI_1_Ygr', idindex='Country')


if __name__ == '__main__':
    unittest.main()
