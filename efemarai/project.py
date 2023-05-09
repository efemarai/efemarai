import hashlib
import os
import zipfile
from time import sleep

import boto3
from appdirs import user_data_dir
from botocore.errorfactory import ClientError

from efemarai.baseline import Baseline
from efemarai.console import console
from efemarai.dataset import Dataset, DatasetFormat
from efemarai.domain import Domain
from efemarai.model import Model, ModelFile, ModelRepository
from efemarai.problem_type import ProblemType
from efemarai.stress_test import StressTest


class Project:
    """
    Provides project related functionality.
    A project is the main building block around which an ML model evaluation is performed.

    A project is an association of a set of ML models, datasets and domains.
    """

    @staticmethod
    def create(
        session, name, description, problem_type, private=False, exists_ok=False
    ):
        """
        Creates a project.

        You should use `ef.Session().create_project(...)` instead.
        """
        if name is None or not ProblemType.has(problem_type):
            raise ValueError(
                f"Project name '{name}' is not specified or problem type '{problem_type}' not recognized."
            )

        existing_project = next((p for p in session.projects if p.name == name), None)
        if existing_project is not None:
            if exists_ok:
                return existing_project
            raise ValueError(f"Project {name} already exists and exists_ok=False.")

        response = session._put(
            "api/project",
            json={
                "name": name,
                "description": description,
                "problem_type": problem_type,
                "private": private,
            },
        )
        return Project(session, response["id"], name, description, problem_type)

    def __init__(self, session, id, name, description, problem_type):
        self._session = session
        self.id = id
        self.name = name  #: (str) Name of the project.
        self.description = description  #: (str) Description of the project.
        self.problem_type = ProblemType(
            problem_type
        )  #: (:class:`efemarai.problem_type.ProblemType`) Project problem type.

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  name={self.name}"
        res += f"\n  description={self.description}"
        res += f"\n  problem_type={self.problem_type}"
        res += "\n)"
        return res

    def _get(self, endpoint, json=None, params=None):
        return self._session._get(endpoint, json, params, self.id)

    def _post(self, endpoint, json=None, params=None):
        return self._session._post(endpoint, json, params, self.id)

    def _put(self, endpoint, json=None, params=None):
        return self._session._put(endpoint, json, params, self.id)

    def _delete(self, endpoint, json=None, params=None):
        return self._session._delete(endpoint, json, params, self.id)

    def _upload(self, from_url, endpoint, verbose=True):
        return self._session._upload(from_url, endpoint, self.id, verbose)

    @property
    def models(self):
        """Returns a list of the models associated with the project."""
        return [
            Model(
                project=self,
                id=model["id"],
                name=model["name"],
                description=model.get("description"),
                version=model.get("version"),
                repository=ModelRepository(
                    url=model["repository"]["url"],
                    branch=model["repository"].get("branch"),
                    hash=model["repository"].get("hash"),
                ),
                files=[
                    ModelFile(
                        name=f["name"],
                        url=f["url"],
                        upload=f["upload"],
                        hash_code=f["hash"],
                    )
                    for f in model["files"]
                ],
            )
            for model in self._get("api/models")
        ]

    def model(self, model) -> Model:
        """Returns the model specified by the name.

        Args:
            model (str): A model identifier. It can either be the name of the model or its id.

        Returns:
            :class:`efemarai.model.Model`: A model object or `None` if it doesn't exist.
        """
        model = next((m for m in self.models if model in (m.name, m.id)), None)
        return model

    def create_model(
        self,
        name,
        description=None,
        repository=None,
        files=None,
        exists_ok=False,
        **kwargs,
    ) -> Model:
        """Creates a model.

        Args:
            name (str): Name of the model.
            description (str, optional): Free text description.
            repository (dict, optional): Model repository info. See `ef.model.ModelRepository`.
            files (list[dict]): Files needed to load the model e.g. weights or config
                files. See `ef.model.ModelFile`.
            exists_ok (bool, optional): If the current named model exist, should it be
                returned or an exception raised.

        Raises:
            ValueError: If the model exists and `exists_ok` is set to `False`.

        Returns:
            :class:`efemarai.model.Model`: The created model.
        """
        existing_model = next((m for m in self.models if m.name == name), None)
        if existing_model is not None:
            if exists_ok:
                return existing_model
            raise ValueError(f"Model {name} already exists and exists_ok=False.")

        model = Model.create(self, name, description, repository, files)
        return model

    def push_model(
        self,
        name,
        description=None,
        repository=None,
        files=None,
        **kwargs,
    ) -> Model:
        """Pushes a new model version or creates a model, if there is no such model with this name.

        Args:
            name (str): Name of the model.
            description (str, optional): Free text description.
            repository (dict, optional): Model repository info. See `ef.model.ModelRepository`.
            files (list[dict]): Files needed to load the model e.g. weights or config
                files. See `ef.model.ModelFile`.

        Returns:
            :class:`efemarai.model.Model`: The created model.
        """
        existing_model = next((m for m in self.models if m.name == name), None)
        if existing_model is not None:
            model = Model.push(
                project=self,
                name=name,
                description=description,
                repository=repository,
                files=files,
                existing_model=existing_model,
            )
            return model
        return Model.create(
            project=self,
            name=name,
            description=description,
            repository=repository,
            files=files,
        )

    @property
    def datasets(self):
        """
        Returns a list of the datasets associated with the project.

        Returns:
            list[:class:`efemarai.dataset.Dataset`]: A list of available datasets.
        """
        return [
            Dataset(
                project=self,
                name=dataset["name"],
                stage=dataset["stage"],
                id=dataset["id"],
                format=dataset["format"],
                data_url=dataset.get("data_url"),
                annotations_url=dataset.get("annotations_url"),
                state=dataset["states"][-1]["name"],
                classes=Dataset._parse_classes(dataset["classes"]),
            )
            for dataset in self._get("api/datasets")
        ]

    def dataset(self, dataset) -> Dataset:
        """
        Returns the dataset specified by the name or id.

        Args:
            dataset (str): A dataset identifier. It can either be the name of the dataset or its id.

        Returns:
            :class:`efemarai.dataset.Dataset`: A dataset object or `None` if it doesn't exist.
        """
        dataset = next((d for d in self.datasets if dataset in (d.name, d.id)), None)
        return dataset

    def create_dataset(
        self,
        name,
        stage,
        format=None,
        data_url=None,
        annotations_url=None,
        credentials=None,
        upload=False,
        num_datapoints=None,
        mask_generation=None,
        exists_ok=False,
        min_asset_area=15,
        **kwargs,
    ) -> Dataset:
        """Creates a dataset.

        Args:
            name (str): A dataset name.
            format (str): A format of the dataset. Possible formats are specified under `ef.dataset.DatasetFormat`.
            stage (str): Dataset stage - one of [`train`, `validation`, `test`].
            data_url (str): A url to fetch data. Can be local or remote.
            annotations_url (str, optional): A url that specifies the annotation file. Can be local or remote.
            credentials (str, optional): If the dataset if remote, Specify the `access token` or `{username}:{password}` to access.
            upload (bool, optional): Should the data be uploaded to Efemarai? This would make evaluation and testing much faster.
            num_datapoints (int, optional): Should a subset of the datapoints be loaded as a dataset? Specify the number of dataponts. Useful for quick testing. Use `None` to load all.
            mask_generation (str): What strategy should be used for mask generation.
            exists_ok (bool, optional): If the current named dataset exist, should it be returned or an exception raised.
            min_asset_area (int, optional): Assets with area smaller than this will be ignored. Default value of 15.

        Raises:
            ValueError: If the dataset exists and `exists_ok` is set to `False`.

        Returns:
            :class:`efemarai.dataset.Dataset`: The created dataset.
        """
        existing_dataset = next((d for d in self.datasets if d.name == name), None)
        if existing_dataset is not None:
            if exists_ok:
                return existing_dataset
            raise ValueError(f"Dataset {name} already exists and exists_ok=False.")

        if isinstance(format, str):
            format = DatasetFormat(format)

        if format == DatasetFormat.Custom:
            dataset = Dataset.create(
                project=self,
                name=name,
                format=format,
                stage=stage,
                data_url=data_url,
                annotations_url=annotations_url,
                credentials=credentials,
                upload=upload,
                num_datapoints=num_datapoints,
                mask_generation=mask_generation,
                min_asset_area=min_asset_area,
            )
        elif format == DatasetFormat.COCO:
            from efemarai.formats.dataset_coco import create_coco_dataset

            dataset = create_coco_dataset(
                project=self,
                name=name,
                stage=stage,
                data_url=data_url,
                annotations_url=annotations_url,
                num_datapoints=num_datapoints,
                mask_generation=mask_generation,
                min_asset_area=min_asset_area,
            )
        elif format == DatasetFormat.ImageNet:
            from efemarai.formats.dataset_imagenet import create_imagenet_dataset

            dataset = create_imagenet_dataset(
                project=self,
                name=name,
                stage=stage,
                data_url=data_url,
                num_datapoints=num_datapoints,
            )
        else:
            console.print(
                f":poop: Cannot load dataset with format {format}. Please use our sdk.",
                style="red",
            )
            return None

        return dataset

    @property
    def domains(self):
        """
        Returns a list of the domains associated with the project.

        Returns:
            list [:class:`efemarai.domain.Domain`]: A list of available domains.
        """
        return [
            Domain(
                project=self,
                id=domain["id"],
                name=domain["name"],
                transformations=domain["transformations"],
            )
            for domain in self._get("api/domains")
        ]

    def domain(self, domain) -> Domain:
        """
        Returns the domain specified by the name or id.

        Args:
            domain (str): A domain identifier. It can either be the name of the domain or its id.

        Returns:
            :class:`efemarai.domain.Domain`: A domain object or `None` if it doesn't exist.
        """
        domain = next((d for d in self.domains if domain in (d.name, d.id)), None)
        return domain

    def create_domain(self, name, transformations, exists_ok=False, **kwargs) -> Domain:
        """
        Creates a domain. **The recommended strategy** is to use the UI to build up the domain visually.
        The domain then can be exported to a yaml configuration file for storage and loaded/reused through `ef.Session().load(...)`.

        Args:
            name (str): The name of the domain.
            transformations (dict): A dictionary of the transformations and their params.
            exists_ok (bool, optional): If the current named domain exist, should it be returned or an exception raised.

        Raises:
            ValueError: If the domain exists and `exists_ok` is set to `False`.

        Returns:
            :class:`efemarai.domain.Domain`: The created domain.
        """
        existing_domain = next((d for d in self.domains if d.name == name), None)
        if existing_domain is not None:
            if exists_ok:
                return existing_domain
            raise ValueError(f"Domain {name} already exists and exists_ok=False.")

        domain = Domain.create(self, name, transformations)
        return domain

    @property
    def baselines(self):
        """
        Returns a list of the baselines associated with the project.

        Returns:
            list [:class:`efemarai.baseline.Baseline`]: A list of available baselines.
        """
        return [
            Baseline(
                project=self,
                id=baseline["id"],
                name=baseline["name"],
                model=baseline["model"],
                dataset=baseline["dataset"]["id"],
                state=baseline["states"][-1]["name"],
                state_message=baseline["states"][-1].get("message"),
                reports=baseline["reports"],
            )
            for baseline in self._get("api/baselineRuns")["objects"]
        ]

    def baseline(self, baseline_run) -> Baseline:
        """
        Returns the baseline specified by the name or id.

        Args:
            baseline_run (str): A stress baseline_run identifier. It can either be the name of the stress test or its id.

        Returns:
            :class:`efemarai.baseline.Baseline`: A baseline object or `None` if it doesn't exist.
        """
        baseline_run = next(
            (b for b in self.baselines if baseline_run in (b.name, b.id)), None
        )
        return baseline_run

    def create_baseline(
        self,
        name,
        model,
        dataset,
        **kwargs,
    ) -> Baseline:
        """
        Creates a stress test.

        Args:
            name (str): The name of the baseline
            model (str, :class:`efemarai.model.Model`): The model to stress test.
            domain (str, :class:`efemarai.domain.Domain`): The domain from which to generate samples.
            dataset (str, :class:`efemarai.dataset.Dataset`): The dataset with which to generate samples.

        Returns:
            :class:`efemarai.baseline.Baseline`: A baseline object.
        """
        baseline = Baseline.create(
            project=self,
            name=name,
            model=model,
            dataset=dataset,
        )

        return baseline

    @property
    def stress_tests(self):
        """
        Returns a list of the stress tests associated with the project.

        Returns:
            list [:class:`efemarai.stress_test.StressTest`]: A list of available stress tests.
        """
        return [
            StressTest(
                project=self,
                id=test["id"],
                name=test["name"],
                model=test["model"],
                domain=test["domain"]["id"],
                dataset=test["dataset"]["id"],
                state=test["states"][-1]["name"],
                state_message=test["states"][-1].get("message"),
                reports=test["reports"],
            )
            for test in self._get("api/getRuns")["objects"]
        ]

    def stress_test(self, test) -> StressTest:
        """
        Returns the stress test specified by the name or id.

        Args:
            test (str): A stress test identifier. It can either be the name of the stress test or its id.

        Returns:
            :class:`efemarai.stress_test.StressTest`: A stress test object or `None` if it doesn't exist.
        """
        test = next((t for t in self.stress_tests if test in (t.name, t.id)), None)
        return test

    def create_stress_test(
        self,
        name,
        model,
        domain,
        dataset,
        num_samples=25,
        num_runs=40,
        concurrent_runs=4,
        **kwargs,
    ) -> StressTest:
        """
        Creates a stress test.

        Args:
            name (str): The name of the stress test.
            model (str, :class:`efemarai.model.Model`): The model to stress test.
            domain (str, :class:`efemarai.domain.Domain`): The domain from which to generate samples.
            dataset (str, :class:`efemarai.dataset.Dataset`): The dataset with which to generate samples.
            num_samples (int, optional): Sample count to generate.
            num_runs (int, optional): Number of searches to run.
            concurrent_runs (int, optional): Number of concurrent searches.

        Returns:
            :class:`efemarai.stress_test.StressTest`: A stress test object.
        """
        test = StressTest.create(
            project=self,
            name=name,
            model=model,
            domain=domain,
            dataset=dataset,
            num_samples=num_samples,
            num_runs=num_runs,
            concurrent_runs=concurrent_runs,
        )

        return test

    def delete(self, delete_dependants=False):
        """Deletes a project, including the domains, datasets, models, stress tests. Cannot be undone."""
        self._delete(f"api/project/{self.id}/{delete_dependants}")

    def export_datasets(
        self,
        test_run_ids: list = None,
        min_score=0.0,
        include_dataset=False,
        path=None,
        unzip=True,
        ignore_cache=False,
        export_format=None,
    ):
        """
        Returns the vulnerabilities dataset associated with the stress tests.

        Args:
            test_run_ids (list[str]): Test runs ids for the datasets.
            min_score (float, optional): Minimum score to select samples.
            include_dataset (bool, optional): If the original dataset used for each of the stress tests should be included.
            path (str, optional): Path to the downloading location.
            unzip (bool, optional): If the zip file should be unzipped.
            ignore_cache (bool, optional): Force regeneration of the dataset by ignoring the cache. May lead to slower subsequent calls.
            export_format (str): The format of the output vulnerabilities dataset.

        Returns:
            str: The filename of the resulting object.
        """
        if export_format is None:
            if self.problem_type == ProblemType.Classification:
                export_format = "imagenet"
            elif self.problem_type == ProblemType.ObjectDetection:
                export_format = "coco"
            elif self.problem_type == ProblemType.InstanceSegmentation:
                export_format = "coco"
            elif self.problem_type == ProblemType.Keypoints:
                export_format = "coco"

        if export_format is None:
            console.print(":poop: Unsupported problem type.", style="red")
            return None

        if not ignore_cache:
            name = "vulnerabilities_dataset"
            name += f"_{export_format}_{include_dataset}_{min_score:.3f}"
            cache_name = os.path.join(path, name)
            if os.path.exists(cache_name) or os.path.exists(cache_name + ".zip"):
                return cache_name

        access = self._post(
            "api/exportDataset",
            json={
                "id": test_run_ids,
                "format": export_format,
                "merge": include_dataset,
                "min_score": min_score,
                "async_download": True,
            },
        )

        s3 = boto3.client(
            "s3",
            aws_access_key_id=access["AccessKeyId"],
            aws_secret_access_key=access["SecretAccessKey"],
            aws_session_token=access["SessionToken"],
            endpoint_url=access["Url"],
        )

        test_run_names = " ,".join(
            [self.stress_test(test_id).name for test_id in test_run_ids]
        )
        # TODO: Check if the dataset does not have any samples and exit
        with console.status(f"Generating '{test_run_names}' vulnerabilities dataset"):
            while True:
                try:
                    response = s3.head_object(
                        Bucket=access["Bucket"], Key=access["ObjectKey"]
                    )
                    size = response["ContentLength"]
                    break
                except ClientError:
                    sleep(1)

        with self._session._progress_bar(verbose=True) as progress:
            task = progress.add_task("Downloading dataset ", total=float(size))

            def callback(num_bytes):
                return progress.advance(task, num_bytes)

            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, os.path.basename(access["ObjectKey"]))

            s3.download_file(
                access["Bucket"], access["ObjectKey"], filename, Callback=callback
            )

        if unzip:
            with console.status("Unzipping dataset"):
                dirname = os.path.splitext(filename)[0]
                with zipfile.ZipFile(filename, "r") as f:
                    f.extractall(dirname)

                os.remove(filename)

                filename = dirname

        console.print(
            (
                f":heavy_check_mark: Downloaded '{test_run_names}' "
                f"vulnerabilities dataset to \n  {filename}"
            ),
            style="green",
        )

        return filename

    def download_joint_vulnerabilities_datasets(
        self,
        test_run_ids: list = None,
        min_score=0.0,
        include_dataset=False,
        path=None,
        unzip=True,
        ignore_cache=False,
        export_format=None,
    ):
        """
        Returns a joint dataset of all specified stress tests.

        Args:
            test_run_ids (list[str]): Ids of the test runs to download.
            min_score (float, optional): Minimum score to select samples.
            include_dataset (bool, optional): If the original dataset used for each stress test should be included.
            path (str, optional): Path to the downloading location.
            unzip (bool, optional): If the zip file should be unzipped.
            ignore_cache (bool, optional): Force regeneration of the dataset by ignoring the cache. May lead to slower subsequent calls.
            export_format (str): The format of the output vulnerabilities dataset.

        Returns:
            str: The filename of the resulting object.
        """
        stress_tests = [self.stress_test(test_id) for test_id in test_run_ids]
        for stress_test in stress_tests:
            if not stress_test.finished:
                console.print(
                    (
                        ":warning: Cannot export vulnerabilities "
                        "dataset as stress test is still running"
                    ),
                    style="yellow",
                )
                return None

        if len({stress_test.dataset.id for stress_test in stress_tests}) > 1:
            console.print(
                ":poop: Cannot merge different original datasets."
                "Please specify stress tests with the same original dataset.",
                style="red",
            )
            return None

        if path is None:
            path = user_data_dir(appname="efemarai")

        test_run_ids.sort()
        ids_hash = hashlib.sha1("".join(test_run_ids).encode("utf-8")).hexdigest()[:24]
        path = os.path.join(path, ids_hash)

        filename = self.export_datasets(
            test_run_ids=test_run_ids,
            min_score=min_score,
            include_dataset=include_dataset,
            path=path,
            unzip=unzip,
            ignore_cache=ignore_cache,
            export_format=export_format,
        )
        return filename
