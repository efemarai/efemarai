from efemarai.job_state import JobState
from efemarai.model import Model


class Baseline:
    """Provides Baseline related functionality."""

    @staticmethod
    def create(
        project,
        name,
        model,
        dataset,
    ):
        """Create a baseline. A more convenient way is to use :func:`project.create_baseline`"""
        if isinstance(model, str):
            model = project.model(model)

        if isinstance(dataset, str):
            dataset = project.dataset(dataset)

        response = project._put(
            "api/baselineRunById",
            json={
                "name": name,
                "model": model.id,
                "dataset": dataset.id,
            },
        )
        return Baseline(
            project,
            response["id"],
            name,
            model,
            dataset,
            "NotStarted",
            None,
            {},
        )

    def __init__(
        self, project, id, name, model, dataset, state, state_message, reports
    ):
        """Create a standard baseline evaluation."""
        self.project = (
            project  #: (:class:`efemarai.project.Project`) Associated project.
        )
        self.id = id
        self.name = name  #: (str) Name of the baseline.

        if isinstance(model, str):
            self._model = None
            self._model_id = model
        elif isinstance(model, dict):
            # TODO: Create a model.reload() to fetch model repository and files
            self._model = Model(
                **model, project=self.project, repository=None, files=None
            )
            self._model_id = self._model.id
        else:
            self._model = model
            self._model_id = self._model.id

        if isinstance(dataset, str):
            self._dataset = None
            self._dataset_id = dataset
        else:
            self._dataset = dataset
            self._dataset_id = dataset.id

        self.state = JobState(
            state
        )  # (:class:`efemarai.stress_test.JobState`) State of the baseline.

        self.state_message = (
            state_message  # (str) Optional message associated with the state
        )

        self._reports = reports

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  name={self.name}"
        res += f"\n  model={self.model.name}"
        res += f"\n  dataset={self.dataset.name}"
        res += f"\n  state={self.state}"
        res += f"\n  state_message={self.state_message}"
        res += f"\n  len(reports)={len(self._reports)}"
        res += "\n)"
        return res

    @property
    def model(self):
        """Returns the model associated with the baseline."""
        if self._model is None:
            self._model = next(
                (m for m in self.project.models if m.id == self._model_id), None
            )
        return self._model

    @property
    def dataset(self):
        """Returns the dataset associated with the baseline."""
        if self._dataset is None:
            self._dataset = next(
                (d for d in self.project.datasets if d.id == self._dataset_id), None
            )
        return self._dataset

    @property
    def finished(self):
        """Returns if the baseline has successfully finished.

        :rtype: bool
        """
        return self.state == JobState.Finished

    @property
    def failed(self):
        """Returns if the baseline has failed.

        :rtype: bool
        """
        return self.state == JobState.Failed

    @property
    def running(self):
        """Returns if the baseline is still running - not failed or finished.

        :rtype: bool
        """
        return self.state not in (JobState.Finished, JobState.Failed)

    def delete(self, delete_dependants=False):
        """Deletes a baseline run. This cannot be undone."""
        self.project._delete(f"api/baselineRunById/{self.id}/{delete_dependants}")

    def reload(self):
        """
        Reloads the baseline *in place* from the remote endpoint and return it.

        Rerturns:
            The updated baseline object.
        """
        response = self.project._get(f"api/baselineRunById/{self.id}")["value"]

        self.name = response["name"]
        self.state = JobState(response["states"][-1]["name"])
        self.state_message = response["states"][-1].get("message")
        self._reports = response.get("reports", {})

        return self
