import re

from efemarai.fields import _decode_ndarray, _encode_ndarray


class Domain:
    """
    Provides domain related functionality. The easiest way to interact with this object is through the UI.
    It can be created through the :class:`efemarai.project.Project.create_domain` method.
    """

    @staticmethod
    def create(project, name, transformations):
        """Create a domain. A more convenient way is to use `project.create_domain(...)`."""
        if name is None or name is None or transformations is None:
            return None

        response = project._put(
            "api/domain",
            json={"name": name, "projectId": project.id},
        )
        domain_id = response["id"]

        response = project._put(
            f"api/domain/{domain_id}/import-flow",
            json={
                "transformations": transformations,
            },
        )

        return Domain(project, domain_id, name, response["transformations"])

    def __init__(self, project, id, name, transformations):
        self.project = (
            project  #: (:class:`efemarai.project.Project`) Associated project.
        )
        self.id = id
        self.name = name  #: (str) Name of the domain
        self.transformations = transformations

    def __call__(self, image):
        return self.apply(image)

    def __repr__(self):
        res = f"{self.__module__}.{self.__class__.__name__}("
        res += f"\n  id={self.id}"
        res += f"\n  name={self.name}"
        res += f"\n  transformations={self.transformations}"
        res += "\n)"
        return res

    def download(self, filename=None) -> str:
        """
        Downalod the domain specifications to a local file.

        Args:
            filename (str, optional): Specify the filename used to store the domain data.

        Returns:
            (str) The filename of the downloaded domain.
        """
        if filename is None:
            # Remove non-ascii and non-alphanumeric characters
            filename = re.sub(r"[^A-Za-z0-9 ]", r"", self.name)
            # Collapse repeating spaces
            filename = re.sub(r"  +", r" ", filename)
            # Replace spaces with dashes and convert to lowercase
            filename = filename.replace(" ", "_").lower()
            filename += ".yaml"

        response = self.project._get(f"api/domain/{self.id}/export")
        with open(filename, "w") as f:
            f.write(response["definition"])

        return filename

    def delete(self, delete_dependants=False):
        """Delete the domain. You cannot delete an object that is used in a stress test or a baseline (delete those first). This cannot be undone."""
        self.project._delete(
            f"api/domain/{self.id}/{delete_dependants}",
        )

    def apply(self, image):
        response = self.project._post(
            "api/generateImageSdk",
            json={
                "domainId": str(self.id),
                "image": _encode_ndarray(image),
                "inputKeyName": "image",
            },
        )

        if "image" not in response:
            return None

        image = _decode_ndarray(response["image"])
        return image
