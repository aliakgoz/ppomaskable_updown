import numpy as np


class SimulationConfig:
    VOLATILITY_LOG_MEAN = -0.4
    VOLATILITY_LOG_SIGMA = 0.6
    DRIFT_MEAN = 0.0
    DRIFT_STD = 2.0
    THETA_LOG_MEAN = 2.0
    THETA_LOG_SIGMA = 1.0
    ETA_LOG_MEAN = 1.0
    ETA_LOG_SIGMA = 1.0

    @staticmethod
    def sample_parameters(batch_size: int):
        sigma_arr = np.random.lognormal(
            mean=SimulationConfig.VOLATILITY_LOG_MEAN,
            sigma=SimulationConfig.VOLATILITY_LOG_SIGMA,
            size=batch_size,
        )
        sigma_arr = np.maximum(sigma_arr, 0.05)

        mu_arr = np.random.normal(
            loc=SimulationConfig.DRIFT_MEAN,
            scale=SimulationConfig.DRIFT_STD,
            size=batch_size,
        )

        theta_arr = np.random.lognormal(
            mean=SimulationConfig.THETA_LOG_MEAN,
            sigma=SimulationConfig.THETA_LOG_SIGMA,
            size=batch_size,
        )

        eta_arr = np.random.lognormal(
            mean=SimulationConfig.ETA_LOG_MEAN,
            sigma=SimulationConfig.ETA_LOG_SIGMA,
            size=batch_size,
        )

        return mu_arr, sigma_arr, theta_arr, eta_arr
