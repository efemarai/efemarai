import os
import re

from setuptools import find_packages, setup


def get_version():
    filename = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "efemarai/__init__.py"
    )
    with open(filename, "r") as f:
        for line in f.read().splitlines():
            match = re.search("^__version__ = [\"'](?P<version>.*)[\"']$", line)
            if match:
                return match.group("version")

        raise RuntimeError("Unable to find version string.")


def get_requirements():
    with open("requirements.txt") as f:
        requirements = [
            line
            for line in f.read().splitlines()
            if not line.startswith("#") and len(line) > 0
        ]
    return requirements


def get_readme():
    with open("README.md") as f:
        readme = f.read()
    return readme


setup(
    name="efemarai",
    version=get_version(),
    description="A CLI and SDK for working with the Efemarai ML testing platform.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="Efemarai",
    author_email="team@efemarai.com",
    url="https://www.efemarai.com/",
    license="MIT license",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["efemarai = efemarai.cli:main", "ef = efemarai.cli:main"]
    },
    install_requires=get_requirements(),
    python_requires=">=3.6",
    zip_safe=False,
)
