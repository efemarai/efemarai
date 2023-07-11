import numpy as np
from rich.table import Table

from efemarai.console import console


class RobustnessTestReport:
    @staticmethod
    def calculate_scores(samples):
        scores = {}

        params = samples.columns.tolist()
        params.remove("baseline_score")
        params.remove("sample_score")
        params.remove("score")
        params.remove("image")

        for param in params:
            scores[param] = {
                "baseline_score": samples.groupby(param)["baseline_score"]
                .mean()
                .mean(),
                "sample_score": samples.groupby(param)["sample_score"].mean().mean(),
                "vulnerability": samples.groupby(param)["score"].mean().mean(),
            }

        return scores

    def __init__(self, samples):
        self.samples = samples
        self.scores = self.calculate_scores(samples)

    def print_vulnerability(self):
        table = Table(title="Robustness Test Report")
        table.add_column("Axis", justify="center")
        table.add_column("Original Failure", justify="center")
        table.add_column("Generated Failure", justify="center")
        table.add_column("Vulnerability", justify="center")

        for param, scores_dict in sorted(
            self.scores.items(),
            key=lambda item: -item[1]["vulnerability"],
        ):
            table.add_row(
                param,
                f"{scores_dict['baseline_score']:.4f}",
                f"{scores_dict['sample_score']:.4f}",
                f"{scores_dict['vulnerability']:.4f}",
            )

        console.print(table)

    def plot(self, filename=None):
        import matplotlib.pyplot as plt

        params = list(self.scores.keys())
        params.sort(key=lambda param: -self.scores[param]["vulnerability"])

        fig, axs = plt.subplots(
            nrows=len(params),
            figsize=(10, 4 * len(params)),
            sharey=True,
        )

        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
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
            ax.set_title(f"{param}: {self.scores[param]['vulnerability']:.4f}")

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
