import os
import re
import tempfile
import urllib
from contextlib import contextmanager

import yaml
from furl import furl

from efemarai.base_checker import BaseChecker
from efemarai.console import console
from efemarai.dataset import DatasetFormat, DatasetStage
from efemarai.runtime_checker import RuntimeChecker

# from efemarai.problem_type import ProblemType


class DefinitionChecker(BaseChecker):
    _allowed_vars = {"datapoints"}

    def __init__(self):
        super().__init__()
        self._status = console.status("Checking...")

    @staticmethod
    def is_path_remote(path):
        return urllib.parse.urlparse(path).scheme in ("http", "https")

    @staticmethod
    def load_definition(filename):
        if not os.path.isfile(filename):
            BaseChecker._error(f"File '{filename}' does not exist")

        with open(filename) as f:
            contents = f.read()
        contents = os.path.expandvars(contents)

        unknown_environment_variables = list(
            re.findall(r"\$\{([a-zA-Z]\w*)\}", contents)
        )

        unknown_environment_variables = list(
            set(unknown_environment_variables) - DefinitionChecker._allowed_vars
        )
        if unknown_environment_variables:
            for match in unknown_environment_variables:
                BaseChecker._error(
                    f"Unknown environment variable '{match}' in '{filename}'"
                )

        return yaml.safe_load(contents)

    def check(
        self,
        definition,
        definition_filename=None,
        check_all=False,
        check_project=False,
        check_datasets=False,
        check_models=False,
        check_domains=False,
        check_runtime=False,
    ):
        if definition_filename is None:
            definition_filename = ""

        try:
            self._status.start()
            if "project" in definition or check_project or check_all:
                self.check_project(definition)

            if "datasets" in definition or check_datasets or check_all:
                self.check_datasets(definition)

            if "models" in definition or check_models or check_all:
                # Model runtime information is resolvable only if it is
                # provided in a file called `efemarai.yaml` since this is
                # what the system looks for when loading a model runtime.
                resolvable_runtime = (
                    os.path.basename(definition_filename) == "efemarai.yaml"
                )
                self.check_models(definition, resolvable_runtime, check_runtime)

            if "domains" in definition or check_domains or check_all:
                self.check_domains(definition)
        except AssertionError:
            self._status.stop()
            return False

        self._status.stop()
        return True

    def check_from_file(
        self,
        definition_filename,
        check_all=False,
        check_project=False,
        check_datasets=False,
        check_models=False,
        check_domains=False,
        check_runtime=False,
    ):
        try:
            definition = self.load_definition(definition_filename)
        except AssertionError:
            return False

        return self.check(
            definition,
            check_all=check_all,
            check_project=check_project,
            check_datasets=check_datasets,
            check_models=check_models,
            check_domains=check_domains,
            check_runtime=check_runtime,
            definition_filename=definition_filename,
        )

    def check_project(self, definition):
        self._status.update("Checking project...")

        project = self._get_required_item(definition, "project")

        # name = self._get_required_item(project, "name", "project")

        # problem_type = self._get_required_item(project, "problem_type", "project")

        # if not ProblemType.has(problem_type):
        #     self._error(f"Unsupported problem type '{problem_type}' (in 'project')")

    def check_datasets(self, definition):
        self._status.update("Checking datasets...")

        datasets = self._get_required_item(definition, "datasets")

        if not isinstance(datasets, list):
            self._error("'datasets' must be an array")

        known_datasets = set()

        for i, dataset in enumerate(datasets):
            parent = f"datasets[{i}]"

            name = self._get_required_item(dataset, "name", parent)

            if name in known_datasets:
                self._error(f"Multiple datasets named '{name}' exist (in 'datasets')")

            known_datasets.add(name)

            format = self._get_required_item(dataset, "format", parent)
            if not DatasetFormat.has(format):
                self._error(f"Unsupported dataset format '{format}' (in '{parent}')")

            stage = self._get_required_item(dataset, "stage", parent)
            if not DatasetStage.has(stage):
                self._error(f"Unsupported dataset stage '{stage}' (in '{parent}')")

            upload = dataset.get("upload", True)

            if upload:
                annotations_url = dataset.get("annotations_url")
                if annotations_url is not None and not os.path.exists(annotations_url):
                    self._error(
                        f"File path '{annotations_url}' does not exist (in '{parent}')"
                    )

                data_url = dataset.get("data_url")
                if data_url is not None and not os.path.exists(data_url):
                    self._error(
                        f"File path '{data_url}' does not exist (in '{parent}')"
                    )

    def check_models(self, definition, resolvable_runtime, check_runtime):
        self._status.update("Checking models...")

        models = self._get_required_item(definition, "models")

        if not isinstance(models, list):
            self._error("'models' must be an array")

        known_models = set()

        for i, model in enumerate(models):
            parent = f"models[{i}]"
            name = self._get_required_item(model, "name", parent)

            if name in known_models:
                self._error(f"Multiple models named '{name}' exist (in 'models')")

            known_models.add(name)

        for i, model in enumerate(models):
            parent = f"models[{i}]"
            self._status.update(f"Checking {parent}...")

            self._check_files(model, parent)

            with self._check_repository(model, parent) as repo_path:
                if model["name"] in {"${model.name}", "$model.name"}:
                    self._warning(f"Skipping default runtime checks (in {parent})")
                    continue

                try:
                    model["runtime"] = self._resolve_runtime(
                        model, repo_path, resolvable_runtime, parent
                    )
                except AssertionError:
                    self._error(f"Unable to find runtime for {parent}")

                if check_runtime:
                    self._status.update(
                        f"Checking {parent}\n  :left_arrow_curving_right: Running model..."
                    )
                    runtime_checker = RuntimeChecker(
                        model,
                        parent,
                        datasets=definition.get("datasets"),
                        project=definition.get("project"),
                        repo_path=repo_path,
                        print_warnings=self._print_warnings,
                    )
                    try:
                        runtime_checker.check()
                    except Exception as e:
                        raise e

    def check_domains(self, definition):
        self._status.update("Checking domains...")

        domains = self._get_required_item(definition, "domains")

        if not isinstance(domains, list):
            self._error("'domains' must be an array")

        known_domains = set()

        for i, domain in enumerate(domains):
            parent = f"domain[{i}]"
            name = self._get_required_item(domain, "name", parent)

            if name in known_domains:
                self._error(f"Multiple models named '{name}' exist (in 'domains')")

            known_domains.add(name)

            _ = self._get_required_item(domain, "transformations", parent)

    @contextmanager
    def _check_repository(self, model, parent):
        repository = model.get("repository", {"url": "."})

        repo_parent = parent + ".repository"

        url = self._get_required_item(repository, "url", repo_parent)

        if DefinitionChecker.is_path_remote(url):
            self._status.update(
                f"Checking {parent}\n  :left_arrow_curving_right: Cloning repo..."
            )

            branch = repository.get("branch")
            hash = repository.get("hash")
            access_token = repository.get("access_token")

            if branch is None and hash is None:
                self._error(f"'branch' or 'hash' must be provided (in '{repo_parent}')")

            # Attempt to import GitPython which checks if a 'git' executable
            # is available and raises an ImportError if it isn't.
            try:
                import git
            except ImportError:
                self._warning(
                    "Skipping remote repo check as no 'git' executable is found."
                )
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    clone_url = furl(url)
                    clone_url.username = access_token
                    clone_url.password = "x-oauth-basic"

                    repo = git.Repo.clone_from(
                        clone_url.tostr(),
                        temp_dir,
                        branch=branch,
                        depth=1,
                        single_branch=True,
                    )
                except Exception:
                    self._error(
                        f"Unable to clone repository at '{url}' (in '{repo_parent}')"
                    )

                if hash is not None:
                    try:
                        repo.commit(hash)
                    except git.exc.GitError:
                        self._error(
                            f"Commit '{hash}' does not exist (in '{repo_parent}')"
                        )

                yield repo.working_dir
        else:
            # Local path - convert to absolute path
            url = os.path.abspath(url)
            if not os.path.isdir(url):
                self._error(
                    f"Repository path '{url}' (in '{repo_parent}') must be a folder."
                )
            # Check if the directory is empty
            if len(os.listdir(url)) == 0:
                self._error(f"Repository path '{url}' (in '{repo_parent}') is empty.")
            yield url

    def _check_files(self, model, parent):
        files = model.get("files", [])
        known_files = set()
        for i, file in enumerate(files):
            file_parent = parent + f".files[{i}]"
            name = self._get_required_item(file, "name", file_parent)

            if name in known_files:
                self._error(f"Multiple files named '{name}' exist (in '{parent}')")

            known_files.add(name)

            url = self._get_required_item(file, "url", parent + f".files[{i}]")

            if file.get("upload", True) and not os.path.exists(url):
                self._error(f"File path '{url}' does not exist (in '{file_parent}')")

    def _resolve_runtime(self, model, repo_path, resolvable_runtime, parent):
        if "runtime" in model:
            if resolvable_runtime:
                return model["runtime"]

            self._warning(
                f"Runtime will not be used - model runtime will be resolved from"
                f" the respective `efemarai.yaml` file (in {parent}.runtime)"
            )

        default_filename = os.path.join(repo_path, "efemarai.yaml")
        if not os.path.exists(default_filename):
            raise AssertionError()

        definition = self.load_definition(default_filename)
        resolved_models = definition.get("models")

        if not resolved_models:
            raise AssertionError()

        for resolved_model in resolved_models:
            if resolved_model["name"] in {
                model["name"],
                "${model.name}",
                "$model.name",
            }:
                runtime = resolved_model.get("runtime")
                if runtime is None:
                    raise AssertionError()
                return runtime

        raise AssertionError()
