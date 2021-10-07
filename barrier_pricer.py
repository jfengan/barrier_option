import math
from scipy.stats import norm


class Barrier(object):
    TAG_DOWN_AND_OUT = "DownAndOut"
    TAG_UP_AND_OUT = "UpAndOut"
    TAG_CALL = "Call"
    TAG_PUT = "Put"

    def __init__(self, spot, strike, r, q, vol, tau, barrier, barrier_type, payoff):
        self.spot = spot
        self.strike = strike
        self.r = r
        self.q = q
        self.vol = vol
        self.tau = tau
        self.barrier = barrier
        assert barrier_type in [Barrier.TAG_UP_AND_OUT, Barrier.TAG_DOWN_AND_OUT], "Unsupported barrier type"
        self.barrier_type = barrier_type
        assert payoff in [Barrier.TAG_CALL, Barrier.TAG_PUT], "Unsupported option payoff"
        self.payoff = payoff
        self.var = self.vol * math.sqrt(self.tau)

    def param_a(self):
        return (self.barrier / self.spot) ** (-1 + (2 * (self.r - self.q) / self.vol ** 2))

    def param_b(self):
        return (self.barrier / self.spot) ** (1 + (2 * (self.r - self.q) / self.vol ** 2))

    def d1(self):
        return (math.log(self.spot / self.strike) + (self.r - self.q + .5 * self.vol ** 2) * self.tau) / self.var

    def d2(self):
        return self.d1() - self.var

    def d3(self):
        return (math.log(self.spot / self.barrier) + (self.r - self.q + .5 * self.vol ** 2) * self.tau) / self.var

    def d4(self):
        return self.d3() - self.var

    def d5(self):
        return (math.log(self.spot / self.barrier) - (self.r - self.q - .5 * self.vol ** 2) * self.tau) / self.var

    def d6(self):
        return self.d5() - self.var

    def d7(self):
        return (math.log(self.spot * self.strike / self.barrier ** 2) -
                (self.r - self.q - .5 * self.vol ** 2) * self.tau) / self.var

    def d8(self):
        return self.d7() - self.var

    def price(self):
        if self.barrier_type == Barrier.TAG_UP_AND_OUT and self.payoff == Barrier.TAG_CALL:
            return self.spot * math.exp(-self.q * self.tau) * (
                    norm.cdf(self.d1()) - norm.cdf(self.d3()) -
                    self.param_b() * (norm.cdf(self.d6()) - norm.cdf(self.d8()))) - \
                   self.strike * math.exp(-self.r * self.tau) * (
                    norm.cdf(self.d2()) - norm.cdf(self.d4()) -
                    self.param_a() * (norm.cdf(self.d5()) - norm.cdf(self.d7()))
                   )
        elif self.barrier_type == Barrier.TAG_DOWN_AND_OUT and self.payoff == Barrier.TAG_CALL:
            if self.barrier > self.strike:
                return self.spot * math.exp(-self.q * self.tau) * (
                        norm.cdf(self.d3()) - self.param_b() * (1 - norm.cdf(self.d6()))) - \
                       self.strike * math.exp(-self.r * self.tau) * (
                        norm.cdf(self.d4()) - self.param_a() * (1 - norm.cdf(self.d5()))
                       )
            else:
                return self.spot * math.exp(-self.q * self.tau) * (
                        norm.cdf(self.d1()) - self.param_b() * (1 - norm.cdf(self.d8()))) - \
                       self.strike * math.exp(-self.r * self.tau) * (
                        norm.cdf(self.d2()) - self.param_a() * (1 - norm.cdf(self.d7()))
                       )

    def delta(self):
        if self.barrier_type == Barrier.TAG_UP_AND_OUT and self.payoff == Barrier.TAG_CALL:
            return 0.
