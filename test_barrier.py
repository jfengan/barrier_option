import math
from barrier_pricer import Barrier
from black_scholes import BlackScholes
from simulator import Simulator


def get_vanilla_call_payoff(x, strike):
    return max(x[-1] - strike, 0)


def test_up_and_out_call(spot, strike, r, q, vol, tau, barrier, dt, time_steps, num_paths):
    def get_uoc_payoff(x, strike, barrier):
        if max(x) >= barrier:
            print(x)
            return 0.
        else:
            return max(0, x[-1] - strike)
    paths = Simulator.simulate_sobol(spot, r, q, vol, dt, num_paths, time_steps)
    uoc_payoff = [get_uoc_payoff(x, strike, barrier) for x in paths]
    vanilla_payoff = [get_vanilla_call_payoff(x, strike) for x in paths]
    del paths
    analytical_uoc_price = Barrier(spot, strike, r, q, vol, tau, barrier, Barrier.TAG_UP_AND_OUT, Barrier.TAG_CALL).price()
    analytical_option_price = BlackScholes(spot, strike, r, q, tau, vol, BlackScholes.TAG_CALL).price()
    print(f"Price for vanilla option: {analytical_option_price}")
    print(f"MC Price for vanilla option: {math.exp(-r * tau) * sum(vanilla_payoff) / len(vanilla_payoff)}")
    print(f"Price for uoc option: {analytical_uoc_price}")
    print(f"MC Price for uoc option: {math.exp(-r * tau) * sum(uoc_payoff) / len(uoc_payoff)}")



if __name__ == "__main__":
    spot = 90
    strike = 100
    r = 0.01
    q = 0.
    vol = 0.1
    tau = 1
    barrier = 120
    dt = 1 / 365
    time_steps = int(tau / dt)
    num_paths = 200000
    test_up_and_out_call(spot, strike, r, q, vol, tau, barrier, dt, time_steps, num_paths)
