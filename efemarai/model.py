import hashlib
import os
import tempfile
from contextlib import contextmanager
from glob import glob
from pathlib import Path
from zipfile import ZipFile

import git
import pathspec

from efemarai.console import console
from efemarai.definition_checker import DefinitionChecker


class ModelRepository:
    def __init__(self, url=None, branch=None, hash=None, access_token=None):
        self.url = url if url is not None else "."
        self.branch = branch
        self.hash = hash
        self.access_token = access_token
        self.files = self.get_all_files() if not self.is_remote else []

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n      url={self.url}"
        res += f"\n      branch={self.branch}"
        res += "\n    )"
        return res

    @property
    def is_remote(self):
        return DefinitionChecker.is_path_remote(self.url)

    def get_all_files(self):
        ignore_files = [
            ignore_file
            for ignore_file in [".gitignore", ".efignore", ".efemaraiignore"]
            if Path.joinpath(Path(self.url), ignore_file).exists()
        ]

        # Do not upload .git folder
        ignore_lines = [".git"]

        # Filter out files based on *ignore specification
        for ignore_file in ignore_files:
            ignore_lines.extend(Path(ignore_file).read_text().splitlines())

        ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)

        files = [
            file
            for file in list(Path(self.url).glob("**/*"))
            if not ignore_spec.match_file(str(file))
        ]

        return files

    @staticmethod
    def calculate_file_hash(filepath):
        return ModelHasher._calculate_file_hash(filepath=filepath)

    def get_last_commit(self, access_token):
        remote_heads = git.cmd.Git().ls_remote(
            self.url.replace(
                "//github",
                f"//{access_token}:x-oauth-basic@github",
            ),
            heads=True,
        )
        return remote_heads.split(f"\trefs/heads/{self.branch}")[0]

    @contextmanager
    def archive(self, name):
        with tempfile.TemporaryDirectory() as dirpath:
            zip_name = os.path.join(dirpath, name)

            with ZipFile(zip_name, "w") as zip:
                for file in self.files:
                    zip.write(file)

            filesize_gb = os.stat(zip_name).st_size / (1024**3)
            if filesize_gb > 1:
                console.print(
                    f":face_with_monocle: "
                    f"Model code is suspiciously large ({filesize_gb:.2f} GB) - "
                    f"consider using '.gitignore' or '.efignore' file",
                    style="orange1",
                )

            yield zip_name


class ModelFile:
    def __init__(self, name, url, upload=True, credentials=None, hash_code=None):
        self.name = name
        self.url = url
        self.upload = upload
        self.credentials = credentials
        self.hash_code = hash_code

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n      name={self.name}"
        res += f"\n      url={self.url}"
        res += f"\n      upload={self.upload}"
        res += f"\n      hash_code={self.hash_code}"
        res += "\n    )"
        return res


class ModelHasher:
    @staticmethod
    def _calculate_file_hash(filepath, hashing_algorithm="sha1"):
        """This function returns the SHA-1 hash
        of the file passed into it."""
        # make a hash object
        if hashing_algorithm == "md5":
            h = hashlib.md5()
        elif hashing_algorithm == "sha1":
            h = hashlib.sha1()

        if str(filepath).startswith("http"):
            return hashlib.sha1(str(filepath).encode("utf-8")).hexdigest()

        if not os.path.exists(filepath):
            console.print(
                f":poop: File/folder under '{filepath}' does not exist.", style="red"
            )
            raise ValueError()

        if os.path.isdir(filepath):
            filenames = [
                filename
                for filename in glob(os.path.join(filepath, "**/*"), recursive=True)
                if os.path.isfile(filename)
            ]
            file_hashes = [ModelHasher._calculate_file_hash(file) for file in filenames]
            h_hash = b"".join(f"{f}".encode("utf-8") for f in file_hashes)
            h.update(h_hash)
            return h.hexdigest()

        # open file for reading in binary mode
        with open(filepath, "rb") as file:
            # loop till the end of the file
            chunk = 0
            while chunk != b"":
                # read only 1024 bytes at a time
                chunk = file.read(1024)
                h.update(chunk)

        # return the hex representation of digest
        return h.hexdigest()


class Model:
    """
    Provides model related functionality.

    It should be created through the :class:`efemarai.project.Project.create_model` method.

    Example:

    .. code-block:: python
        :emphasize-lines: 2

        import efemarai as ef
        ef.Session().project("Name").create_model(...)
    """

    @staticmethod
    def create(project, name, description, repository, files):
        """
        Creates a model.

        You should use :func:`project.create_model` instead.
        """
        if name is None:
            raise ValueError("Missing model name")

        if description is None:
            description = ""

        if repository is None:
            repository = {}

        if files is None:
            files = []

        if not isinstance(repository, ModelRepository):
            repository = ModelRepository(**repository)

        for f in files:
            f["hash_code"] = ModelHasher._calculate_file_hash(filepath=f["url"])

        model_files_hashes = "".join([f["hash_code"] for f in files])

        # Remote repo case
        if repository.is_remote:
            repo_hash = (
                repository.hash
                if repository.hash is not None
                else repository.get_last_commit(repository.access_token)
            )
        else:
            # Local repo case
            if len(repository.files) == 0:
                console.print(
                    f"No files in '{repository.url}'. "
                    "Confirm path and ignore rules in '.gitignore', '.efignore'."
                )
                raise ValueError()

            repo_hash = "".join(
                [
                    ModelHasher._calculate_file_hash(filepath=i)
                    for i in repository.files
                    if i.is_file()
                ]
            )

        version = hashlib.sha1(
            (repo_hash + model_files_hashes).encode("utf-8")
        ).hexdigest()  # Model hash

        files = [ModelFile(**f) if not isinstance(f, ModelFile) else f for f in files]

        response = project._put(
            "api/model",
            json={
                "name": name,
                "description": description,
                "version": version,
                "repository": {
                    "url": repository.url,
                    "branch": repository.branch,
                    "hash": repository.hash,
                    "access_token": repository.access_token,
                },
                "files": [
                    {
                        "name": f.name,
                        "url": f.url,
                        "upload": f.upload,
                        "credentials": f.credentials,
                        "hash": f.hash_code,
                    }
                    for f in files
                ],
            },
        )
        model_id = response["id"]

        if not repository.is_remote:
            base_name = "model_code.zip"
            endpoint = f"api/modelCode/{model_id}/upload"
            with repository.archive(name=base_name) as archive_name:
                project._upload(archive_name, endpoint)

            project._post(endpoint, json={"archive": base_name})

        for f in files:
            if f.upload:
                project._upload(f.url, f"api/model/{model_id}/upload")

        return Model(project, model_id, name, description, version, repository, files)

    @staticmethod
    def push(project, name, description, repository, files, existing_model):
        """
        Pushes a new model version.

        You should use :func:`project.push_model` instead.
        """
        if name is None:
            raise ValueError("Missing model name")

        if description is None:
            description = ""

        if repository is None:
            repository = {}

        if files is None:
            files = []

        if not isinstance(repository, ModelRepository):
            repository = ModelRepository(**repository)

        for f in files:
            f["hash_code"] = ModelHasher._calculate_file_hash(filepath=f["url"])

        model_files_hashes = "".join([f["hash_code"] for f in files])

        if repository.is_remote:
            # Remote repo
            repo_hash = (
                repository.hash
                if repository.hash is not None
                else repository.get_last_commit(repository.access_token)
            )
        else:
            # Local repo
            if len(repository.files) == 0:
                console.print(
                    f"No files in '{repository.url}'. "
                    "Confirm path and ignore rules in '.gitignore', '.efignore'."
                )
                raise ValueError()

            repo_hash = "".join(
                [
                    ModelHasher._calculate_file_hash(filepath=f)
                    for f in repository.files
                    if f.is_file()
                ]
            )

        version = hashlib.sha1(
            (repo_hash + model_files_hashes).encode("utf-8")
        ).hexdigest()  # Model hash

        # Do not create a new model version if there are no changes
        if version == existing_model.version:
            console.print(
                ":face_with_monocle: "
                "There are were no changes found in the files. Skipping...",
                style="orange1",
            )
            return existing_model

        files = [ModelFile(**f) if not isinstance(f, ModelFile) else f for f in files]
        files = Model._point_to_existing_model_files(
            new_model_files=files, existing_model=existing_model
        )
        response = project._put(
            "api/model",
            json={
                "name": name,
                "description": description,
                "version": version,
                "repository": {
                    "url": repository.url,
                    "branch": repository.branch,
                    "hash": repository.hash,
                    "access_token": repository.access_token,
                },
                "files": [
                    {
                        "name": f.name,
                        "url": f.url,
                        "upload": f.upload,
                        "credentials": f.credentials,
                        "hash": f.hash_code,
                    }
                    for f in files
                ],
            },
        )
        model_id = response["id"]

        if not repository.is_remote:
            base_name = "model_code.zip"
            endpoint = f"api/modelCode/{model_id}/upload"
            with repository.archive(name=base_name) as archive_name:
                project._upload(archive_name, endpoint)

            project._post(endpoint, json={"archive": base_name})

        for f in files:
            if f.upload:
                project._upload(f.url, f"api/model/{model_id}/upload")

        return Model(project, model_id, name, description, version, repository, files)

    def __init__(self, project, id, name, description, version, repository, files):
        self.project = project
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.repository = repository
        self.files = files

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  name={self.name}"
        res += f"\n  description={self.description}"
        res += f"\n  version={self.version}"
        res += f"\n  repository={self.repository}"
        res += f"\n  files={self.files}"
        res += "\n)"
        return res

    def delete(self, delete_dependants=False):
        """
        Deletes the model.

        You cannot delete an object that is used in a stress test or a baseline
        (delete those first). Deletion cannot be undone.
        """
        self.project._delete(
            f"api/model/{self.id}/{delete_dependants}",
        )

    @staticmethod
    def _point_to_existing_model_files(new_model_files, existing_model):
        """
        Checks if the new model files are different from the old ones.
        Points the url to the old ones if they are the same.
        """
        for f in new_model_files:
            if f.hash_code in [f.hash_code for f in existing_model.files]:
                f.url = [
                    file.url
                    for file in existing_model.files
                    if file.hash_code == f.hash_code
                ][0]
                f.upload = False

        return new_model_files

    def versions(self):
        """Returns all model versions."""
        return self.project._get(f"api/modelVersions/{self.id}")
