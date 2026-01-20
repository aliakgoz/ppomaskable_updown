import numpy as np
from scipy.stats import norm


class SimulationEngine:
    def __init__(self):
        self.T_seconds = 900
        self.dt = 1.0
        self.SECONDS_IN_YEAR = 365.0 * 24.0 * 3600.0

    def generate_batch(self, n_paths, mu, sigma, theta=None, eta=None, S0=100.0):
        num_steps = self.T_seconds
        dt_year = self.dt / self.SECONDS_IN_YEAR
        sqrt_dt = np.sqrt(dt_year)

        if theta is None:
            theta = np.zeros((n_paths, 1))
        if eta is None:
            eta = np.zeros((n_paths, 1))

        current_mu = mu
        Z_S = np.random.normal(0, 1, size=(n_paths, num_steps))
        Z_mu = np.random.normal(0, 1, size=(n_paths, num_steps))

        log_returns = np.zeros((n_paths, num_steps))

        for t in range(num_steps):
            drift_part = (current_mu - 0.5 * sigma**2) * dt_year
            shock_part = sigma * sqrt_dt * Z_S[:, t : t + 1]
            log_returns[:, t : t + 1] = drift_part + shock_part

            d_mu = theta * (mu - current_mu) * dt_year + eta * sqrt_dt * Z_mu[:, t : t + 1]
            current_mu = current_mu + d_mu

        log_path = np.cumsum(log_returns, axis=1)
        S_t_log = np.log(S0) + log_path
        S_t = np.exp(S_t_log)

        K = S0
        time_elapsed = np.arange(1, num_steps + 1)
        time_remaining_seconds = self.T_seconds - time_elapsed
        time_remaining_years = np.maximum(time_remaining_seconds, 1e-6) / self.SECONDS_IN_YEAR
        T_minus_t = time_remaining_years.reshape(1, -1)

        numerator = np.log(S_t / K) - 0.5 * (sigma**2) * T_minus_t
        denominator = sigma * np.sqrt(T_minus_t)
        d2 = numerator / denominator

        probabilities = norm.cdf(d2)
        final_S = S_t[:, -1]
        final_payoff = np.where(final_S > K, 1.0, 0.0)
        probabilities[:, -1] = final_payoff
        probabilities[:, 0] = 0.5

        return probabilities
