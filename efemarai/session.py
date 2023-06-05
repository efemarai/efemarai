import ntpath
import os
import posixpath
import re
from datetime import datetime
from glob import glob

import boto3
import requests
import validators
import yaml
from appdirs import user_config_dir
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeRemainingColumn
from rich.prompt import Confirm, Prompt
from smart_open import smart_open

from efemarai.console import console
from efemarai.definition_checker import DefinitionChecker
from efemarai.problem_type import ProblemType
from efemarai.project import Project


class Session:
    """
    Session related functionality. A session is the way to interact with the Efemarai endpoint.
    All commands use the active session to communicate and perform the needed actions.
    """

    DEFAULT_URL = "https://ci.efemarai.com"

    @staticmethod
    def _fetch_token(url, username, password):
        try:
            response = requests.post(
                url + "auth/sdk-token/",
                json={"username": username, "password": password},
            )

            if not response.ok:
                console.print(":poop: Invalid username or password", style="red")
                return None

            return response.json()["token"]

        except requests.exceptions.ConnectionError:
            console.print(f":poop: Unreachable URL {url}", style="red")
            return None

    @staticmethod
    def _user_setup(config_file=None):
        console.print(":rocket:  [bold]Welcome to Efemarai![/bold]\n", style="#00a9ff")

        existing_config = Session._read_config(config_file)
        if existing_config is None:
            console.print("Let's set up things quickly.")
        else:
            console.print("URL: ", existing_config["url"])
            console.print("User: ", existing_config["username"])
            console.print()
            if not Confirm.ask(
                "Do you want to overwrite existing config?", default=False
            ):
                return None

        console.print()

        username = Prompt.ask("Username")
        password = Prompt.ask("Password", password=True)

        while True:
            url = Prompt.ask("Platform URL", default=Session.DEFAULT_URL)

            if not re.match(r"^https?://", url):
                url = "https://" + url

            if not url.endswith("/"):
                url += "/"

            if validators.url(url):
                break

            console.print("[prompt.invalid]:poop: Invald URL")

        return Session._setup(url, username, password, config_file)

    @staticmethod
    def _setup(url, username, password, config_file=None):
        token = Session._fetch_token(url, username, password)

        if token is None:
            return None

        config = {"username": username, "url": url, "token": token}

        if config_file is None:
            config_dir = user_config_dir(appname="efemarai")
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "config.yaml")

        console.print(f":gear: Saving config in {config_file}", style="green")

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config

    @staticmethod
    def _read_config(config_file=None):
        if config_file is None:
            config_file = os.path.join(
                user_config_dir(appname="efemarai"), "config.yaml"
            )

        if not os.path.isfile(config_file):
            return None

        return Session._load_config_file(config_file)

    @staticmethod
    def _load_config_file(filename):
        checker = DefinitionChecker()
        return checker.load_definition(filename)

    @staticmethod
    def _progress_bar(verbose):
        return Progress(
            SpinnerColumn(style="green"),
            "[progress.description]{task.description}",
            BarColumn(complete_style="default"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            transient=True,
            disable=not verbose,
        )

    def __init__(self, token=None, url=None, config_file=None):
        self.token = token
        self.url = url

        if self.token is None or self.url is None:
            config = self._read_config(config_file)

            if config is None:
                config = self._user_setup(config_file)

            self.token = config["token"]
            self.url = config["url"]

        self._access_requests = {}

    def load(self, filename, exists_ok=False, project_only=False) -> dict:
        """Load a configuration file for creating projects, models, datasets and/or domains.

        Args:
            filename (str): A configuration file.
            exists_ok (bool, optional): If any of the components defined in the configuration file exist, should they raise an exception or return the already existing element.
            project_only (bool, optional): Load only the specified project.

        Riases:
            ValueError: If any of the components exist and `exists_ok` is set to `False`.

        Returns:
            dict: A dictionary of the potentially loaded projects, datasets, domains, models.

        Example:

        .. code-block:: python

            import efemarai as ef
            result = ef.Session().load("efemarai.yaml")
            print(result)
        """
        config = self._load_config_file(filename)

        project = None
        if "project" in config:
            project = self.create_project(**config["project"], exists_ok=exists_ok)

        if project_only:
            return {"project": project}

        models = [
            project.create_model(**model_config, exists_ok=exists_ok)
            for model_config in config.get("models", [])
        ]

        datasets = [
            project.create_dataset(**dataset_config, exists_ok=exists_ok)
            for dataset_config in config.get("datasets", [])
        ]
        domains = []
        for domain_config in config.get("domains", []):
            if "file" in domain_config:
                domain_config = self._load_config_file(domain_config["file"])["domain"]

            domains.append(project.create_domain(**domain_config, exists_ok=exists_ok))

        return {
            "project": project,
            "datasets": datasets,
            "domains": domains,
            "models": models,
        }

    @property
    def projects(self):
        """
        Returns the list of projects.

        Returns:
            list [:class:`efemarai.project.Project`]: A list of the available projects.
        """
        return [
            Project(
                self,
                project["id"],
                project["name"],
                project.get("description", ""),
                project["problem_type"],
            )
            for project in self._get("api/projects")
        ]

    def project(self, name):
        """
        Returns a project specified by the name.

        Args:
            name (str): A project identifier. It can either be the name of the project or its id.

        Returns:
            :class:`efemarai.project.Project`: A project object or `None` if it doesn't exist.

        Example:

        .. code-block:: python

            import efemarai as ef
            project = ef.Session().project("Name")
            print(project)
        """
        project = next((p for p in self.projects if name in (p.name, p.id)), None)
        return project

    def create_project(
        self,
        name,
        description=None,
        problem_type=ProblemType.Custom.value,
        private=False,
        exists_ok=False,
        **kwargs,
    ):
        """
        Creates a project.

        Args:
            name (str): Name of the project.
            description (str, optional): Description of the project.
            problem_type (str, :class:`efemarai.problem_type.ProblemType`): Problem type of the project.
            exists_ok (bool, optional): If the current named project exist, should it be returned or an exception raised.

        Raises:
            ValueError: If the project exists and `exists_ok` is set to `False`.

        Returns:
            :class:`efemarai.project.Project`: A project object.
        """
        project = Project.create(
            self, name, description, problem_type, private, exists_ok
        )
        return project

    def _get(self, endpoint, json=None, params=None, project_id=None, verbose=True):
        return self._make_request(
            requests.get, endpoint, json, params, project_id, verbose
        )

    def _post(self, endpoint, json=None, params=None, project_id=None, verbose=True):
        return self._make_request(
            requests.post, endpoint, json, params, project_id, verbose
        )

    def _put(self, endpoint, json=None, params=None, project_id=None, verbose=True):
        return self._make_request(
            requests.put, endpoint, json, params, project_id, verbose
        )

    def _delete(self, endpoint, json=None, params=None, project_id=None, verbose=True):
        return self._make_request(
            requests.delete, endpoint, json, params, project_id, verbose
        )

    def _make_request(
        self, method, endpoint, json=None, params=None, project_id=None, verbose=True
    ):
        url = self.url
        if not url.endswith("/") and not endpoint.startswith("/"):
            url += "/"
        url += endpoint

        try:
            headers = {}
            if self.token is not None and self.token != "":
                headers.update({"Authorization": f"Token {self.token}"})
            if project_id is not None:
                headers.update({"ProjectId": project_id})

            response = method(
                url,
                headers=headers,
                json=json,
                params=params,
            )
        except requests.exceptions.ConnectionError as e:
            if verbose:
                console.print(
                    f":poop: It looks you cannot reach us at {self.url}. If you think you should, send us an email.",
                    style="RED",
                )
            raise e

        if not response.ok:
            try:
                message = response.json()["message"]
            except Exception:
                message = response.text
            if verbose:
                console.print(
                    f":poop: [{response.status_code} {response.reason}] {message}",
                    style="RED",
                )

            response.raise_for_status()

        try:
            return response.json()
        except Exception:
            return None

    def _upload(self, from_url, endpoint, project_id, verbose=True):
        if os.path.isdir(from_url):
            self._upload_directory(from_url, endpoint, project_id, verbose)
        else:
            self._upload_file(from_url, endpoint, project_id, verbose)

    def _upload_file(self, from_url, endpoint, project_id, verbose=True):
        object_key = os.path.basename(from_url)

        with self._progress_bar(verbose) as progress:
            task = progress.add_task(
                f"Uploading '{object_key}' ",
                total=float(os.path.getsize(from_url)),
            )

            def callback(num_bytes):
                return progress.advance(task, num_bytes)

            s3, bucket, prefix = self._get_access(endpoint, project_id)
            with smart_open(from_url, "rb") as f:
                s3.upload_fileobj(f, bucket, prefix + object_key, Callback=callback)

        if verbose:
            console.print(f":heavy_check_mark: Uploaded '{object_key}'", style="green")

    def _upload_directory(self, from_url, endpoint, project_id, verbose=True):
        filenames = [
            filename
            for filename in glob(os.path.join(from_url, "**/*"), recursive=True)
            if os.path.isfile(filename)
        ]

        dirname = os.path.basename(os.path.normpath(from_url))

        s3, bucket, prefix = self._get_access(endpoint, project_id)

        with self._progress_bar(verbose) as progress:
            task = progress.add_task(
                f"Uploading '{dirname}' ", total=float(len(filenames))
            )
            for filename in filenames:
                object_key = os.path.join(
                    dirname, os.path.relpath(filename, start=from_url)
                ).replace(ntpath.sep, posixpath.sep)

                with smart_open(filename, "rb") as f:
                    s3.upload_fileobj(f, bucket, prefix + object_key)

                progress.advance(task)

        if verbose:
            console.print(f":heavy_check_mark: Uploaded '{from_url}'", style="green")

    def _get_access(self, endpoint, project_id):
        access = self._access_requests.get(endpoint)

        if access is None or access["Expiration"] <= datetime.now():
            access = self._get(endpoint, project_id=project_id)
            access["Expiration"] = datetime.strptime(
                access["Expiration"], "%Y-%m-%dT%H:%M:%SZ"
            )
            self._access_requests[endpoint] = access

        s3 = boto3.client(
            "s3",
            aws_access_key_id=access["AccessKeyId"],
            aws_secret_access_key=access["SecretAccessKey"],
            aws_session_token=access["SessionToken"],
            endpoint_url=access["Url"],
        )

        return s3, access["Bucket"], access["Prefix"]
