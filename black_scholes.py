# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:14:58 2021

@author: Administrator
"""

import math
from scipy.stats import norm


class BlackScholes:
    def __init__(self, spot, strike, r, q, tau, vol):
        self.spot = spot
        self.strike = strike
        self.r = r
        self.q = q
        self.tau = tau
        self.vol = vol

    def d1(self):
        return (math.log(self.spot / self.strike) + (self.r - self.q + 1 / 2 * self.vol ** 2) * self.tau) / \
               (self.vol * math.sqrt(self.tau))

    def d2(self):
        return self.d1() - self.vol * math.sqrt(self.tau)

    def price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.spot * norm.cdf(d1) * math.exp(-self.q * self.tau) - self.strike * norm.cdf(d2) * math.exp(-self.r * self.tau)
