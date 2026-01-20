import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from simulation_engine import SimulationEngine
from simulation_config import SimulationConfig


class PolymarketDataGenerator:
    def __init__(self):
        self.engine = SimulationEngine()

    def yield_batches(self, batch_size=1024):
        while True:
            mu, sigma, theta, eta = SimulationConfig.sample_parameters(batch_size)

            mu_r = mu.reshape(-1, 1)
            sigma_r = sigma.reshape(-1, 1)
            theta_r = theta.reshape(-1, 1)
            eta_r = eta.reshape(-1, 1)

            batch = self.engine.generate_batch(
                n_paths=batch_size,
                mu=mu_r,
                sigma=sigma_r,
                theta=theta_r,
                eta=eta_r,
            )

            yield batch, {"mu": mu, "sigma": sigma, "theta": theta, "eta": eta}

    def plot_samples(self, num_samples: int = 5, out_dir: str = "plots/data_generator") -> None:
        if plt is None:
            raise ImportError("matplotlib is required for plotting.")
        batch, meta = next(self.yield_batches(batch_size=num_samples))
        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_samples):
            series = batch[i]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series, color="#1f77b4")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(
                f"Sample {i} | mu={meta['mu'][i]:.2f} sigma={meta['sigma'][i]:.2f} "
                f"theta={meta['theta'][i]:.2f} eta={meta['eta'][i]:.2f}"
            )
            ax.set_xlabel("t")
            ax.set_ylabel("p")
            fig.tight_layout()
            fig.savefig(f"{out_dir}/sample_{i}.png")
            plt.close(fig)


if __name__ == "__main__":
    gen = PolymarketDataGenerator()
    data_stream = gen.yield_batches(batch_size=10)
    batch, meta = next(data_stream)

    print("Batch Shape:", batch.shape)
    print("Sample Mu:", meta["mu"][:3])
    print("Sample Sigma:", meta["sigma"][:3])
    print("First Series End Value:", batch[0, -1])

    try:
        gen.plot_samples(num_samples=3)
        print("Saved sample plots to plots/data_generator/")
    except ImportError:
        print("matplotlib not installed; skipping plots.")
