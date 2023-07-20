import json
import os
import re

from appdirs import user_data_dir

from efemarai.console import console
from efemarai.dataset import Dataset
from efemarai.job_state import JobState
from efemarai.model import Model


class StressTest:
    """
    Provides stress test related functionality.
    It can be created through a :class:`efemarai.project.Project.create_stress_test` method.

    Example:

    .. code-block:: python
        :emphasize-lines: 2,5,7

        import efemarai as ef
        test = ef.Session().project("Name").create_stress_test(...)
        # do something else
        test.reload()
        if test.finished:
            print(f"Report: {test.report}")
            dataset_filename = test.vulnerabilities_dataset(min_score=0.1)

    Example (2):

    .. code-block:: python

        import efemarai as ef
        project = ef.Session().project("Name")
        test = project.stress_test("Test Name")
        test.download_reports()

    Example (3):

    .. code-block:: python

        import efemarai as ef
        project = ef.Session().project("Name")
        test = project.stress_test("Test Name")
        for datapoint in test:
            print(datapoint)
    """

    @staticmethod
    def create(
        project,
        name,
        model,
        domain,
        dataset,
        num_samples,
        num_runs,
        concurrent_runs,
    ):
        """Create a stress test. A more convenient way is to use :func:`project.create_stress_test`"""
        if isinstance(model, str):
            model = project.model(model)

        if isinstance(dataset, str):
            dataset = project.dataset(dataset)

        if isinstance(domain, str):
            domain = project.domain(domain)

        response = project._put(
            "api/stressTest",
            json={
                "name": name,
                "model": model.id,
                "dataset": dataset.id,
                "domain": domain.id,
                "project": project.id,
                "samples_per_run": num_samples,
                "runs_count": num_runs,
                "concurrent_runs": concurrent_runs,
            },
        )
        return StressTest(
            project,
            response["id"],
            name,
            model,
            domain,
            dataset,
            "NotStarted",
            None,
            {},
        )

    def __init__(
        self,
        project,
        id,
        name,
        model,
        domain,
        dataset,
        state,
        state_message,
        reports,
        count=None,
    ):
        self.project = (
            project  #: (:class:`efemarai.project.Project`) Associated project.
        )
        self.id = id
        self.name = name  #: (str) Name of the stress test.

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

        if isinstance(domain, str):
            self._domain = None
            self._domain_id = domain
        else:
            self._domain = domain
            self._domain_id = domain.id

        if isinstance(dataset, str):
            self._dataset = None
            self._dataset_id = dataset
        else:
            self._dataset = dataset
            self._dataset_id = dataset.id

        self.state = JobState(
            state
        )  # (:class:`efemarai.stress_test.JobState`) State of the stress test.

        self.state_message = (
            state_message  # (str) Optional message associated with the state
        )

        self._reports = reports
        self.count = count

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  name={self.name}"
        res += f"\n  model={self.model.name}"
        res += f"\n  domain={self.domain.name}"
        res += f"\n  dataset={self.dataset.name}"
        res += f"\n  state={self.state}"
        res += f"\n  state_message={self.state_message}"
        res += f"\n  len(reports)={len(self.reports)}"
        res += "\n)"
        return res

    def __get_data(self, skip, limit, min_score=-1):
        response = self.project._post(
            "api/testRunIter",
            json={
                "testRunId": self.id,
                "skip": skip,
                "limit": limit,
                "minScore": min_score,
            },
        )

        datapoints = []
        for data in response["objects"]:
            datapoint = Dataset.deserialize_datapoint(data)
            datapoints.append(datapoint)

        return datapoints, response["count"]

    def __len__(self):
        if self.count is None:
            _, datapoints_count = self.__get_data(skip=0, limit=1)
            self.count = datapoints_count

        return self.count

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Note: Currently ignoring the step in the slice
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            datapoints, _ = self.__get_data(skip=start, limit=stop - start)
            return datapoints
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(
                    f"The index ({key}) is out of range (size: {len(self)})."
                )
            datapoints, _ = self.__get_data(skip=key, limit=1)
            return datapoints[0]
        raise TypeError(f"Invalid argument type ({type(key)}).")

    @property
    def reports(self):
        """Returns the stress test reports. This can be slow as is potentially fetching a large object."""
        if not self._reports:
            response = self.project._get(
                "api/getTestRun",
                params={"testRunId": self.id},
            )
            self._reports = response["reports"]

        return self._reports

    @property
    def model(self):
        """Returns the model associated with the stress test."""
        if self._model is None:
            self._model = next(
                (m for m in self.project.models if m.id == self._model_id), None
            )
        return self._model

    @property
    def domain(self):
        """Returns the domain associated with the stress test."""
        if self._domain is None:
            self._domain = next(
                (d for d in self.project.domains if d.id == self._domain_id), None
            )
        return self._domain

    @property
    def dataset(self):
        """Returns the dataset associated with the stress test."""
        if self._dataset is None:
            self._dataset = next(
                (d for d in self.project.datasets if d.id == self._dataset_id), None
            )
        return self._dataset

    @property
    def finished(self):
        """Returns if the stress test has successfully finished.

        :rtype: bool
        """
        return self.state == JobState.Finished

    @property
    def failed(self):
        """Returns if the stress test has failed.

        :rtype: bool
        """
        return self.state == JobState.Failed

    @property
    def running(self):
        """Returns if the stress test is still running - not failed or finished.

        :rtype: bool
        """
        return self.state not in (JobState.Finished, JobState.Failed)

    def delete(self):
        """Deletes a stress test. This cannot be undone."""
        self.project._delete("api/stressTest?id=" + self.id)

    def reload(self):
        """
        Reloads the stress test *in place* from the remote endpoint and return it.

        Rerturns:
            The updated stress test object.
        """
        response = self.project._get("api/getTestRun", params={"testRunId": self.id})

        self.state = JobState(response["states"][-1]["name"])
        self.state_message = response["states"][-1].get("message")
        self._reports = response.get("reports", {})

        return self

    def download_reports(self, filename=None):
        """
        Download any generated reports.

        Args:
            filename (str): Specify the filename used to store the report data.

        Returns:
            str: The filename of the downloaded report.
        """
        if filename is None:
            # Remove non-ascii and non-alphanumeric characters
            filename = re.sub(r"[^A-Za-z0-9 ]", r"", self.name)
            # Collapse repeating spaces
            filename = re.sub(r"  +", r" ", filename)
            # Replace spaces with dashes and convert to lowercase
            filename = filename.replace(" ", "_").lower()
            filename += ".json"

        with open(filename, "w") as f:
            json.dump(self.reports, f)

        return filename

    def vulnerabilities_dataset(
        self,
        min_score=0.0,
        include_dataset=False,
        path=None,
        unzip=True,
        ignore_cache=False,
        export_format=None,
    ):
        """
        Returns the vulnerabilities dataset associated with the stress test.

        Args:
            min_score (float, optional): Minimum score to select samples.
            include_dataset (bool, optional): If the original dataset used for the stress test should be included.
            path (str, optional): Path to the downloading location.
            unzip (bool, optional): If the zip file should be unzipped.
            ignore_cache (bool, optional): Force regeneration of the dataset by ignoring the cache. May lead to slower subsequent calls.
            export_format (str): The format of the output vulnerabilities dataset.

        Returns:
            str: The filename of the resulting object.
        """
        if not self.finished:
            console.print(
                (
                    ":warning: Cannot export vulnerabilities "
                    "dataset as stress test is still running"
                ),
                style="yellow",
            )
            return None

        if path is None:
            path = user_data_dir(appname="efemarai")

        path = os.path.join(path, self.id)

        filename = self.project.export_datasets(
            test_run_ids=[self.id],
            min_score=min_score,
            include_dataset=include_dataset,
            path=path,
            unzip=unzip,
            ignore_cache=ignore_cache,
            export_format=export_format,
        )
        return filename
