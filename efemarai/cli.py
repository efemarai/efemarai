import os
import sys
import time

import click
import cv2
from appdirs import user_config_dir
from bson import ObjectId
from click_aliases import ClickAliasedGroup
from rich.prompt import Confirm
from rich.table import Table

import efemarai as ef
from efemarai.console import console
from efemarai.definition_checker import DefinitionChecker

checker = DefinitionChecker()

default_yaml = "efemarai.yaml"


@click.group()
@click.version_option(ef.__version__)
def main():
    """Efemarai CLI."""
    pass


@main.command()
@click.option("-c", "--config", help="Optional configuration file.")
@click.option(
    "--noinput", is_flag=True, default=False, help="Avoid user input prompts."
)
@click.option("--username", help="Efemarai username.")
@click.option("--url", help="Efemarai URL.")
def init(config, noinput, username, url):
    """Initialize Efemarai."""
    if noinput:
        ef.Session._setup(url, username, password=os.environ.get("EFEMARAI_PASSWORD"))
    else:
        ef.Session._user_setup(config_file=config)


@main.command()
@click.option(
    "-c",
    "--config",
    default=None,
    help=f"Efemarai config filename (Default: '{os.path.join(user_config_dir(appname='efemarai'), 'config.yaml')}')",
)
def status(config):
    """Check the status of the Efemarai system."""
    try:
        config = ef.Session()._read_config(config)
        console.print(
            f"Checking [bold]{config['username']}[/bold] @ {config['url']}",
        )
        # Check status by getting a list of projects
        _ = ef.Session().projects
        console.print(":heavy_check_mark: Status: [bold]Success[/bold]", style="green")
    except Exception:
        console.print(":poop: Status: [bold]Fail[/bold]", style="red")
        sys.exit(1)


@main.command()
@click.argument("definition-file", required=True)
@click.option(
    "-a",
    "--all",
    "check_all",
    default=False,
    is_flag=True,
    help="Perform all checks on a complete project definition including models, datasets and domains.",
)
@click.option(
    "--runtime-check/--no-runtime-check",
    "check_runtime",
    default=False,  # TODO: Remove runtime_check or create dummy datapoin
    help="Check if model can be run successfully.",
)
@click.option(
    "--warnings/--no-warnings",
    default=True,
    help="Print warnings.",
)
def check(definition_file, check_all, check_runtime, warnings):
    """Check if the YAML definition of a project, dataset, model or domain is correct.

    DEFINITION_FILE is the YAML definition file to be checked.
    """
    if definition_file == ".":
        definition_file = default_yaml

    checker.print_warnings(warnings)

    if not checker.check_from_file(
        definition_file, check_all=check_all, check_runtime=check_runtime
    ):
        sys.exit(1)

    console.print(":heavy_check_mark: [bold]No issues detected[/bold]", style="green")


@main.group(cls=ClickAliasedGroup)
def project():
    """Manage projects."""
    pass


@project.command("create")
@click.argument("definition-file", required=True)
@click.option(
    "--exists-ok/--exists-not-ok",
    default=False,
    help="Skip project and its models, datasets, domains if they already exists.",
)
@click.option(
    "-w",
    "--wait",
    default=False,
    is_flag=True,
    help="Wait for any created datasets to be loaded.",
)
@click.option("-v", "--verbose", count=True, help="Print resulting model.")
@click.option(
    "-p",
    "--project-only",
    default=False,
    is_flag=True,
    help="Create just the project specified in the defintion file.",
)
@click.option(
    "--runtime-check/--no-runtime-check",
    "check_runtime",
    default=False,  # TODO: Remove runtime_check or create dummy datapoint
    help="Check if any defined models can be run successfully.",
)
@click.option(
    "--warnings/--no-warnings",
    default=True,
    help="Print warnings when checking the project definition.",
)
def project_create(
    definition_file,
    exists_ok,
    wait,
    verbose,
    project_only,
    check_runtime,
    warnings,
):
    """Create a project following the specified configuration file.

    definition_file (str): YAML file containing project definition."""
    if definition_file == ".":
        definition_file = default_yaml

    checker.print_warnings(warnings)

    if not checker.check_from_file(
        definition_file, check_project=True, check_runtime=check_runtime
    ):
        sys.exit(1)

    result = ef.Session().load(
        definition_file, exists_ok=exists_ok, project_only=project_only
    )

    if verbose:
        console.print(result)

    if wait:
        _wait_for_runs(result["datasets"])


@project.command("list", aliases=["ls"])
def project_list():
    """Lists the projects associated with the current user."""
    table = Table(box=None)
    _ = [table.add_column(x) for x in ["Id", "Name", "Problem Type"]]
    for m in ef.Session().projects:
        table.add_row(m.id, m.name, str(m.problem_type))
    console.print(table)


@project.command("delete")
@click.argument("project", required=True)
@click.option("-y", "--yes", default=False, is_flag=True, help="Confirm deletion.")
def project_delete(project, yes):
    """Delete the specified project."""
    if project == ".":
        project = default_yaml

    if project.endswith((".yaml", ".yml")):
        project = _get_project(efemarai_file=project, must_exist=False)
    else:
        project = ef.Session().project(project)

    if project is None:
        return

    if yes or Confirm.ask(
        f"Do you want to delete project [bold]{project.name}[/bold] including all stress tests, models, datasets and domains?",
        default=False,
    ):
        project.delete(delete_dependants=True)


@main.group(cls=ClickAliasedGroup)
def model():
    """Manage models."""
    pass


@model.command("list", aliases=["ls"])
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def model_list(file):
    """Lists the models in the current project."""
    project = _get_project(efemarai_file=file)

    table = Table(box=None)
    _ = [table.add_column(x) for x in ["Id", "Name"]]
    for m in project.models:
        table.add_row(m.id, m.name)
    console.print(table)


@model.command("log")
@click.argument("model", required=True)
def model_version_list(model):
    """Lists the model versions of a particular model / definition file."""
    if model.endswith((".yaml", ".yml", ".")):
        if model.endswith("."):
            model = default_yaml
        project = _get_project(efemarai_file=model)
        models_yaml = [m["name"] for m in checker.load_definition(model)["models"]]
        models = [model for model in project.models if model.name in models_yaml]
    else:
        project = _get_project(efemarai_file=default_yaml)
        models = [m for m in project.models if model in (m.name, m.id)]

    model_versions = [model.versions() for model in models]

    table = Table(box=None)
    _ = [table.add_column(x) for x in ["Id", "Name", "Description", "Version"]]
    for model_v in model_versions:
        for m in model_v["objects"]:
            table.add_row(m["id"], m["name"], m["description"], m["version"][:7])
    console.print(table)


@model.command("create")
@click.argument("definition-file", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option("-v", "--verbose", count=True, help="Print created models.")
@click.option(
    "--exists-ok/--exists-not-ok",
    default=False,
    help="Skip model if it already exists.",
)
@click.option(
    "--runtime-check/--no-runtime-check",
    "check_runtime",
    default=False,  # TODO: Remove runtime_check or create dummy datapoin
    help="Check if the model can be run successfully.",
)
@click.option(
    "--warnings/--no-warnings",
    default=True,
    help="Print warnings when checking the model definition.",
)
def model_create(definition_file, file, verbose, exists_ok, check_runtime, warnings):
    """Create a model in the current project."""
    if definition_file == ".":
        definition_file = default_yaml

    checker.print_warnings(warnings)

    definition = checker.load_definition(definition_file)

    if not checker.check(
        definition,
        definition_filename=definition_file,
        check_models=True,
        check_runtime=check_runtime,
    ):
        sys.exit(1)

    project = _get_project(definition=definition, efemarai_file=file)

    for model_definition in definition["models"]:
        if model_definition["name"] == "${model.name}":
            continue

        try:
            model = project.create_model(**model_definition, exists_ok=exists_ok)

            if verbose:
                console.print(model)

            console.print(model.id)

        except Exception as e:
            console.print(e, style="red")
            sys.exit(1)


@model.command("push")
@click.argument("definition-file", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option("-v", "--verbose", count=True, help="Print created models.")
@click.option(
    "--runtime-check/--no-runtime-check",
    "check_runtime",
    default=False,  # TODO: Remove runtime_check or create dummy datapoin
    help="Check if the model can be run successfully.",
)
@click.option(
    "--warnings/--no-warnings",
    default=True,
    help="Print warnings when checking the model definition.",
)
@click.option(
    "-m",
    "--message",
    help="Model version message.",
)
def model_push(definition_file, file, verbose, check_runtime, warnings, message):
    """Pushes a model version in the current project."""
    if definition_file == ".":
        definition_file = default_yaml

    checker.print_warnings(warnings)

    definition = checker.load_definition(definition_file)

    if not checker.check(
        definition,
        definition_filename=definition_file,
        check_models=True,
        check_runtime=check_runtime,
    ):
        sys.exit(1)

    project = _get_project(definition=definition, efemarai_file=file)

    for model_definition in definition["models"]:
        if model_definition["name"] == "${model.name}":
            continue

        if message is not None:
            model_definition["description"] = message

        try:
            model = project.push_model(**model_definition)

            if verbose:
                console.print(model)

            console.print(model.id)
        except Exception as e:
            console.print(e, style="red")
            sys.exit(1)


@model.command("delete")
@click.argument("model", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-d",
    "--delete-dependants",
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete all entities that depend on the model",
)
def model_delete(model, file, delete_dependants):
    """Delete a model from the current project.

    MODEL - the name or ID of the model."""
    project = _get_project(efemarai_file=file)
    model_name = model
    model = project.model(model)
    if not model:
        console.print(
            f":poop: Project '{project.name}' does not have model '{model_name}'",
            style="red",
        )
        sys.exit(1)

    model.delete(delete_dependants)


@main.group(cls=ClickAliasedGroup)
def domain():
    """Manage domains."""
    pass


@domain.command("list", aliases=["ls"])
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def domain_list(file):
    """Lists the domains in the current project."""
    project = _get_project(efemarai_file=file)

    table = Table(box=None)
    _ = [table.add_column(x) for x in ["Id", "Name"]]
    for d in project.domains:
        table.add_row(d.id, d.name)
    console.print(table)


@domain.command("create")
@click.argument("definition-file", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option("-v", "--verbose", count=True, help="Print created domains.")
@click.option(
    "--exists-ok/--exists-not-ok",
    default=False,
    help="Skip domain if it already exists.",
)
@click.option(
    "--warnings/--no-warnings",
    default=True,
    help="Print warnings when checking the domain definition.",
)
def domain_create(definition_file, file, verbose, exists_ok, warnings):
    """Create a domain in the current project.

    definition_file (str): YAML file containing domain definition."""
    checker.print_warnings(warnings)

    definition = checker.load_definition(definition_file)

    if not checker.check(
        definition,
        definition_filename=definition_file,
        check_domains=True,
    ):
        sys.exit(1)

    project = _get_project(definition=definition, efemarai_file=file)

    for domain_definition in definition["domains"]:
        try:
            domain = project.create_domain(**domain_definition, exists_ok=exists_ok)

            if verbose:
                console.print(domain)

            console.print(domain.id)

        except Exception as e:
            console.print(e, style="red")
            sys.exit(1)


@domain.command("delete")
@click.argument("domain", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-d",
    "--delete-dependants",
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete all entities that depend on the domain",
)
def domain_delete(domain, file, delete_dependants):
    """Delete a domain from the current project.

    DOMAIN - the name or ID of the domain."""
    project = _get_project(efemarai_file=file)

    if _check_for_multiple_entities(project.domains, domain):
        console.print("There are multiple domains with the given name:\n")
        domains = [t for t in project.domains if t.name == domain]
        _print_table(domains)
        console.print(
            f"\nRun the command with a specific domain id: [bold green]$ efemarai domain delete {domains[-1].id}",
        )
        sys.exit(1)

    domain_name = domain
    domain = project.domain(domain)
    if not domain:
        console.print(f":poop: Domain '{domain_name}' does not exist.", style="red")
        sys.exit(1)
    domain.delete(delete_dependants)


@domain.command("download")
@click.argument("domain", required=True)
@click.option("-o", "--output", default=None, help="Optional domain output file.")
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def domain_download(domain, output, file):
    """Download a domain.

    DOMAIN - the name of the domain."""
    project = _get_project(efemarai_file=file)

    domain = project.domain(domain)
    filename = domain.download(filename=output)
    console.print(
        (f":heavy_check_mark: Downloaded '{domain.name}' reports to: \n  {filename}"),
        style="green",
    )


@domain.command("apply")
@click.argument("domain_name", required=True)
@click.argument("image", required=True)
@click.option(
    "-o", "--output", default="output.png", help="Optional domain output file."
)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def domain_apply(domain_name, image, output, file):
    with console.status("Applying domain transformations...", spinner_style="green"):
        project = _get_project(efemarai_file=file)

        domain = project.domain(domain_name)
        if not domain:
            console.print(f":poop: Domain '{domain_name}' does not exist.", style="red")
            sys.exit(1)

        image = cv2.imread(image)

        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        transformed_image = domain.apply(image)
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

        if transformed_image is not None:
            dirname = os.path.dirname(output)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            cv2.imwrite(output, transformed_image)


@main.group(cls=ClickAliasedGroup)
def dataset():
    """Manage datasets."""
    pass


@dataset.command("list", aliases=["ls"])
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def dataset_list(file):
    """Lists the datasets in the current project."""
    project = _get_project(efemarai_file=file)

    table = Table(box=None)

    for x in ["Id", "Name", "Status"]:
        table.add_column(x)

    for dataset in project.datasets:
        if dataset.finished:
            status = "Loaded"
        elif dataset.failed:
            status = "Failed"
        else:
            status = "Loading"

        table.add_row(dataset.id, dataset.name, status)

    console.print(table)


@dataset.command("create")
@click.argument("definition_file", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-w",
    "--wait",
    default=False,
    is_flag=True,
    help="Wait for dataset to be loaded.",
)
@click.option("-v", "--verbose", count=True, help="Print created datasets.")
@click.option(
    "--exists-ok/--exists-not-ok",
    default=False,
    help="Skip dataset if it already exists.",
)
@click.option(
    "--warnings/--no-warnings",
    default=True,
    help="Print warnings when checking the dataset definition.",
)
def dataset_create(definition_file, file, wait, verbose, exists_ok, warnings):
    """Create a dataset in the current project.

    definition_file (str): YAML file containing dataset definition."""
    checker.print_warnings(warnings)

    if definition_file == ".":
        definition_file = default_yaml

    definition = checker.load_definition(definition_file)

    if not checker.check(
        definition,
        definition_filename=definition_file,
        check_datasets=True,
    ):
        sys.exit(1)

    project = _get_project(definition=definition, efemarai_file=file)

    datasets = []
    for dataset_definition in definition["datasets"]:
        try:
            dataset = project.create_dataset(**dataset_definition, exists_ok=exists_ok)
            datasets.append(dataset)

            if verbose:
                console.print(dataset)

            console.print(dataset.id)

        except Exception as e:
            console.print(e, style="red")
            sys.exit(1)

    if not wait:
        return

    _wait_for_runs(datasets)


@dataset.command("delete")
@click.argument("dataset", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-d",
    "--delete-dependants",
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete all entities that depend on the dataset",
)
def dataset_delete(dataset, file, delete_dependants):
    """Delete a dataset from the current project.

    DATASET - the name or ID of the dataset."""
    project = _get_project(efemarai_file=file)

    if _check_for_multiple_entities(project.datasets, dataset):
        console.print("There are multiple datasets with the given name:\n")
        datasets = [t for t in project.datasets if t.name == dataset]
        _print_table(datasets)
        console.print(
            f"\nRun the command with a specific dataset id: [bold green]$ efemarai dataset delete {datasets[-1].id}",
        )
        sys.exit(1)

    dataset_name = dataset
    dataset = project.dataset(dataset)
    if dataset is None:
        console.print(f":poop: Dataset '{dataset_name}' does not exist.", style="red")
        sys.exit(1)

    dataset.delete(delete_dependants)


@dataset.command("download")
@click.argument("dataset", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "--format",
    required=True,
    help="Format of the downloaded dataset.",
)
@click.option(
    "-p",
    "--path",
    default=".",
    help="The directory where to download the files.",
)
@click.option(
    "-n",
    "--num-samples",
    type=int,
    default=None,
    help="Number of samples to download (Default is all).",
)
@click.option(
    "--unzip/--no-unzip",
    is_flag=True,
    show_default=True,
    default=True,
    help="Do not unzip the downloaded dataset.",
)
@click.option(
    "--ignore-cache",
    is_flag=True,
    default=False,
    help="Force regeneration of the files. May lead to slower subsequent calls.",
)
def dataset_download(dataset, file, format, path, num_samples, unzip, ignore_cache):
    """Download a dataset from the current project.

    DATASET - the name or ID of the dataset."""
    project = _get_project(efemarai_file=file)

    if _check_for_multiple_entities(project.datasets, dataset):
        console.print("There are multiple datasets with the given name:\n")
        datasets = [t for t in project.datasets if t.name == dataset]
        _print_table(datasets)
        console.print(
            f"\nRun the command with a specific dataset id: [bold green]$ efemarai dataset download {datasets[-1].id}",
        )
        sys.exit(1)

    dataset_name = dataset
    dataset = project.dataset(dataset)
    if not dataset:
        console.print(f":poop: Dataset '{dataset_name}' does not exist.", style="red")
        sys.exit(1)

    dataset.download(
        dataset_format=format,
        path=path,
        num_samples=num_samples,
        unzip=unzip,
        ignore_cache=ignore_cache,
    )


@dataset.command("enhance")
@click.argument("dataset", required=True)
@click.argument("domain", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option("--new-name", default=None, help="New name of the dataset.")
@click.option(
    "-s",
    "--samples-per-datapoint",
    type=int,
    default=1,
    help="The number of samples to generate per datapoint.",
)
@click.option(
    "-w",
    "--wait",
    default=False,
    is_flag=True,
    help="Wait for dataset enhancing to finish.",
)
def dataset_enhance(dataset, domain, new_name, samples_per_datapoint, file, wait):
    """Enhance a dataset by applying a domain from the current project."""
    project = _get_project(efemarai_file=file)

    dataset_name = dataset
    dataset = project.dataset(dataset)
    if not dataset:
        console.print(f":poop: Dataset '{dataset_name}' does not exist.", style="red")
        sys.exit(1)

    domain_name = domain
    domain = project.domain(domain)
    if not domain:
        console.print(f":poop: Domain '{domain_name}' does not exist.", style="red")
        sys.exit(1)

    enhanced_dataset = dataset.enhance(domain, samples_per_datapoint, new_name)

    if not wait:
        return

    _wait_for_runs([enhanced_dataset])


@main.group(cls=ClickAliasedGroup)
def baseline():
    """Manage baselines."""
    pass


@baseline.command("list", aliases=["ls"])
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def baseline_list(file):
    """Lists the baselines in the current project."""
    project = _get_project(efemarai_file=file)

    _print_table(project.baselines)


@baseline.command("run")
@click.argument("definition-file", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-w", "--wait", default=False, is_flag=True, help="Wait for baseline to finish."
)
@click.option(
    "-s",
    "--sequential",
    default=False,
    is_flag=True,
    help="Wait for current baseline to finish before starting next.",
)
@click.option("-v", "--verbose", count=True, help="Print created baselines.")
def baseline_run(definition_file, file, wait, sequential, verbose):
    """Run a baseline.

    definition_file (str): YAML file containing baseline definition."""
    create_run(
        definition_file,
        file,
        wait,
        sequential,
        verbose,
        create_attr="create_baseline",
        key_name="baselines",
        url_name="baselines",
    )


def create_run(
    definition_file, file, wait, sequential, verbose, create_attr, key_name, url_name
):
    if not os.path.exists(definition_file):
        console.print(f":poop: File {definition_file} does not exist")
        sys.exit(1)

    definition = ef.Session._load_config_file(definition_file)

    project = _get_project(definition=definition, efemarai_file=file)

    try:
        run_definitions = definition[key_name]
    except KeyError:
        console.print(
            f":poop: Key '{key_name}' not found in definition file: {definition_file}",
            style="red",
        )
        sys.exit(1)

    cfg = ef.Session._read_config()

    runs = []
    for run_definition in run_definitions:
        run = getattr(project, create_attr)(**run_definition)
        runs.append(run)

        if verbose:
            console.print(run)

        console.print(f"{cfg['url']}project/{project.id}/{url_name}/{run.id}")

        if sequential:
            _wait_for_runs([run])

    if not wait:
        return

    _wait_for_runs(runs)


@baseline.command("delete")
@click.argument("baseline", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-d",
    "--delete-dependants",
    is_flag=True,
    show_default=True,
    default=False,
    help="Delete all entities that depend on the baseline",
)
def baseline_delete(baseline, file, delete_dependants):
    """Delete a stress baseline from the current project.

    BASELINE - Name, ID or yaml file of the stress baseline."""
    run_delete(
        baseline,
        file,
        delete_dependants=delete_dependants,
        project_runs_list_attr="baselines",
        project_run_getter="baseline",
        key_name="baselines",
    )


def run_delete(
    run, file, delete_dependants, project_runs_list_attr, project_run_getter, key_name
):
    project = _get_project(efemarai_file=file)

    if _check_for_multiple_entities(getattr(project, project_runs_list_attr), run):
        console.print("There are multiple runs with the given name:\n")
        runs = [t for t in getattr(project, project_runs_list_attr) if t.name == run]
        _print_table(runs)
        console.print(
            f"\nRun the command with a specific run id: [bold green]$ efemarai test/baseline delete {runs[-1].id}",
        )
        sys.exit(1)

    run_name = run

    if run.endswith((".yaml", ".yml")):
        run_name = _get_run_name(run, key=key_name)

    run = getattr(project, project_run_getter)(run_name)

    if run is None:
        console.print(f":poop: Run '{run_name}' does not exist.", style="red")
        sys.exit(1)

    if delete_dependants:
        run.delete(delete_dependants)
    else:
        run.delete()


@main.group(cls=ClickAliasedGroup)
def test():
    """Manage stress tests."""
    pass


@test.command("list", aliases=["ls"])
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def test_list(file):
    """Lists the stress tests in the current project."""
    project = _get_project(efemarai_file=file)

    _print_table(project.stress_tests)


@test.command("run")
@click.argument("definition-file", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
@click.option(
    "-w", "--wait", default=False, is_flag=True, help="Wait for stress test to finish."
)
@click.option(
    "-s",
    "--sequential",
    default=False,
    is_flag=True,
    help="Wait for current stress test to finish before starting next.",
)
@click.option("-v", "--verbose", count=True, help="Print created stress tests.")
def test_run(definition_file, file, wait, sequential, verbose):
    """Run a stress test.

    definition_file (str): YAML file containing stress test definition."""
    create_run(
        definition_file,
        file,
        wait,
        sequential,
        verbose,
        create_attr="create_stress_test",
        key_name="tests",
        url_name="runs",
    )


def _wait_for_runs(runs):
    with console.status("Waiting to finish...", spinner_style="green"):
        for run in runs:
            while run.reload().running:
                time.sleep(0.25)

            if run.failed:
                console.print(
                    f":poop: Failed: \n {run} {run.state_message}",
                    style="red",
                )
                sys.exit(1)


@test.command("delete")
@click.argument("test", required=True)
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def test_delete(test, file):
    """Delete a stress test from the current project.

    TEST - Name, ID or yaml file of the stress test."""
    run_delete(
        test,
        file,
        delete_dependants=False,
        project_runs_list_attr="stress_tests",
        project_run_getter="stress_test",
        key_name="tests",
    )


@test.command("download")
@click.argument("tests_names", nargs=-1, required=True)
@click.option(
    "--format",
    required=True,
    help="Format of the downloaded dataset.",
)
@click.option("--min_score", default=0.0, help="Minimum score for the samples.")
@click.option("--include_dataset", default=False, help="Include original test dataset.")
@click.option("--path", default=None, help="Path to the downloaded files.")
@click.option("--unzip", default=True, help="Whether to unzip the resulting file.")
@click.option("--ignore_cache", default=False, help="Ignore local cache.")
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def test_download(
    tests_names, format, min_score, include_dataset, path, unzip, ignore_cache, file
):
    """Download the stress test vulnerabilities dataset.

    TESTS_NAMES - Name, ID or yaml file of the stress test to download."""
    files = file.split(",") if file is not None else [default_yaml]
    projects = [_get_project(efemarai_file=file) for file in files]

    tests = []
    for test_name in tests_names:
        if test_name.endswith((".yaml", ".yml")):
            test_name = _get_run_name(test_name)
        for project in projects:
            tests.append(project.stress_test(test_name))

    for idx, test in enumerate(tests):
        if test is None:
            console.print(
                f":poop: Stress test '{tests_names[idx]}' does not exist.", style="red"
            )
            sys.exit(1)

    projects[0].download_joint_vulnerabilities_datasets(
        test_run_ids=[test.id for test in tests],
        min_score=min_score,
        include_dataset=include_dataset,
        path=path,
        unzip=unzip,
        ignore_cache=ignore_cache,
        export_format=format,
    )


@test.command("reports")
@click.argument("test", required=True)
@click.option("-o", "--output", default=None, help="Optional output file.")
@click.option(
    "-f", "--file", help=f"Name of the Efemarai file (Default is '{default_yaml}')"
)
def test_reports(test, output, file):
    """Export the stress test reports.

    TEST - Name, ID or yaml file of the stress test."""
    test_name = test
    project = _get_project(efemarai_file=file)

    if test.endswith((".yaml", ".yml")):
        test_name = _get_run_name(test)

    test = project.stress_test(test_name)

    if test is None:
        console.print(f":poop: Stress test '{test_name}' does not exist.", style="red")
        sys.exit(1)

    filename = test.download_reports(filename=output)
    console.print(
        (f":heavy_check_mark: Downloaded '{test.name}' reports to: \n  {filename}"),
        style="green",
    )


def _print_table(tests):
    if len(tests) == 0:
        console.print("Empty table.")
        return

    columns_list = ["id", "name", "model", "dataset", "domain", "state"]
    table = Table(box=None)
    for c in columns_list:
        if hasattr(tests[0], c):
            table.add_column(c.capitalize())
    for t in tests:
        row = []
        for c in columns_list:
            if hasattr(t, c):
                obj = getattr(t, c)
                if hasattr(obj, "name"):
                    obj = obj.name
                row.append(str(obj))
        table.add_row(*row)
    console.print(table)


def _check_for_multiple_entities(entities, name):
    return len([x for x in entities if x.name == name]) > 1


def _get_project(definition=None, efemarai_file=None, must_exist=True):
    # Check if passed file is actually a project object id
    if ObjectId.is_valid(efemarai_file):
        project = ef.Session().project(efemarai_file)
        if project is None and must_exist:
            console.print(
                f":poop: Project '{efemarai_file}' does not exist.", style="red"
            )
            sys.exit(1)

        return project

    # Check if definition needs to be overwritten from file
    if definition is not None and (
        "project" not in definition or "name" not in definition["project"]
    ):
        definition = None

    if definition is None or efemarai_file is not None:
        if efemarai_file is None:
            efemarai_file = default_yaml

        if not os.path.exists(efemarai_file):
            console.print(
                f":poop: '{efemarai_file}' does not exist in '{os.getcwd()}'.",
                style="red",
            )
            sys.exit(1)

        definition = ef.Session()._load_config_file(efemarai_file)

    if "project" not in definition:
        console.print(":poop: Missing field 'project'", style="red")
        sys.exit(1)

    if "name" not in definition["project"]:
        console.print(":poop: Missing filed 'name' (in 'project')", style="red")
        sys.exit(1)

    name = definition["project"]["name"]
    project = ef.Session().project(name)
    if project is None and must_exist:
        console.print(f":poop: Project '{name}' does not exist.", style="red")
        sys.exit(1)

    return project


def _get_run_name(file, index=0, key="tests"):
    config = ef.Session()._load_config_file(file)
    return config[key][index]["name"]


if __name__ == "__main__":
    main()
