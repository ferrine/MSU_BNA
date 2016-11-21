# coding=utf-8
from pymc3.models.linear import Glm
import pandas as pd


class FEPanel(Glm):
    def __init__(self, x, y, index, dummy_na=True, intercept=False, labels=None,
                 priors=None, init=None, vars=None, family='normal', name=''):
        if not isinstance(x, pd.DataFrame):
            raise TypeError('Need Pandas DataFrame for x')
        if not isinstance(y, pd.Series):
            raise TypeError('Need Pandas Series for y')
        if not isinstance(index, (tuple, list)):
            index = [index]
        x = pd.get_dummies(
            x, columns=index, drop_first=not intercept, dummy_na=dummy_na
        )   # type: pd.DataFrame
        is_dummy = lambda s: any(s.startswith('%s_' % l) for l in index)
        self.dummies = list(filter(is_dummy, x.columns))
        self.not_dummies = list(set(x.columns) - set(self.dummies))
        new_priors = dict.fromkeys(
            self.dummies, self.default_intercept_prior
        )
        if priors is None:
            priors = dict()
        new_priors.update(priors)
        super(FEPanel, self).__init__(
            x, y, intercept, labels,
            new_priors, init, vars, family, name
        )

    @property
    def dummies_vars(self):
        return [self[v] for v in self.dummies]

    @property
    def not_dummies_vars(self):
        return [self[v] for v in self.not_dummies]
