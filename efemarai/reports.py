import numpy as np
from rich.table import Table

from efemarai.console import console


class RobustnessTestReport:
    @staticmethod
    def calculate_vulnerability(samples):
        vulnerability = {}

        params = samples.columns.tolist()
        params.remove("score")
        params.remove("image")

        for param in params:
            vulnerability[param] = samples.groupby(param)["score"].mean().mean()

        return vulnerability

    def __init__(self, samples):
        self.samples = samples
        self.vulnerability = self.calculate_vulnerability(samples)

    def print_vulnerability(self):
        table = Table(title="Robustness Test Report")
        table.add_column("Axis", justify="center")
        table.add_column("Vulnerability", justify="center")

        for param, sensitivity in sorted(
            self.vulnerability.items(),
            key=lambda item: -item[1],
        ):
            table.add_row(param, f"{sensitivity:.4f}")

        console.print(table)

    def plot(self, filename=None):
        import matplotlib.pyplot as plt

        params = list(self.vulnerability.keys())
        params.sort(key=lambda param: -self.vulnerability[param])

        fig, axs = plt.subplots(
            nrows=len(params),
            figsize=(10, 4 * len(params)),
            sharey=True,
        )

        axs[0].set_ylabel("Vulnerability")

        for param, ax in zip(params, axs):
            data = self.samples.groupby(param)["score"].apply(np.array)

            ticks = data.index.to_numpy()

            elements = ax.violinplot(
                data.values,
                ticks,
                widths=0.12 / len(ticks),
                showextrema=True,
                showmeans=True,
                showmedians=False,
            )
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_title(f"{param}: {self.vulnerability[param]:.4f}")

            for name in ("cbars", "cmins", "cmaxes", "cmeans"):  # "cmedians",
                elements[name].set_edgecolor("#00a9ff")
                elements[name].set_linewidth(1)

            for violin in elements["bodies"]:
                violin.set_facecolor("#00a9ff")
                violin.set_edgecolor("#00a9ff")
                violin.set_linewidth(0)
                violin.set_alpha(0.5)

        fig.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(f"{filename}")
            console.print(f":heavy_check_mark: Report plot saved as '{filename}'")
