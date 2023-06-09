import functools
import multiprocessing as mp
import uuid
from collections import defaultdict

from rich.progress import Progress, track

import efemarai as ef
from efemarai.console import console


def test_robustness(
    dataset,
    model,
    domain,
    dataset_format,
    output_format,
    input_format=None,
    transform=None,
    target_transform=None,
    transforms=None,
    hooks=None,
    class_ids=None,
    class_names=None,
    dataset_index_to_id=None,
    num_search_steps=100,
    remap_class_ids=None,
):
    """
    Stress test a model and estimate its vulnerability w.r.t. an operational domain.

    Args:
        dataset: list[tuple]
            List of (input, target) used for stress testing.

        model: Callable
            Model inference function mapping inputs to predictions.

        domain: efemarai.metamorph.Domain
            Operational domain specifying the allowed transformations and their ranges.
            Use a predefined one in `efemarai.domains` or create a new one at `ci.efemarai.com`.

        dataset_format: `efemarai.spec`
            Specification of the dataset format. Standard dataset formats are listed
            in `efemarai.formats`. For custom datasets, provide your own `efemarai.spec`
            transforming dataset samples to `efemarai.Datapoint`.

        output_format: `efemarai.spec`
            Specification of the model input format. Standard input formats are listed
            in `efemarai.formats`. For custom models, provide your own `efemarai.spec`
            transforming a model output to a list of `efemarai.fields`.

        input_format: `efemarai.spec`=None
            Specification of the model input format. Standard input formats are listed
            in `efemarai.formats`. For custom models, provide your own `efemarai.spec`
            transforming an `efemarai.Datapoint` to a model input.

        transform: Callable=None
            User specified transformation applied to the inputs of a data sample.

        target_transform: Callable=None
            User specified transformation applied to the targets of a data sample.

        transforms: Callable=None
            User specified transformation applied both to the inputs and targets of a
            data sample.

        hooks: list[Callable]=None
            A list of hooks called after every search step iteration as follows:
                hook(
                    original_datapoint: efemarai.Datapoint,
                    transformed_datapoint: efemarai.Datapoint,
                    model_output: List[efemarai.fields],
                    original_datapoint_loss: dict,
                    transformed_datapoint_loss: dict,
                    vulnerability: float,
                )

        class_ids: list[int]=None
            IDs of the classes in the dataset. If not provided this informatio will be extracted
            from the dataset, but requires a pass over the entire dataset.

        dataset_index_to_id: dict[int, int]=None
            Optional dictionary mapping a dataset index to dataset sample id.

        num_search_steps: int=100,
            Number of iterations searching for failures within the operation domain.

        remap_class_ids: dict[int, int]=None,
            Dictionary with model class_ids to remap to dataset class ids.

    Returns:

        An `efemarai.RobsutnessTestReport` containing vulnerability per domain axis
        as well as info about all searched elements of the operational domain.
    """

    # Check if full version is installed
    try:
        import pandas as pd

        from efemarai.metamorph.loss.loss_functions import aggregate_loss, failure_loss
    except ImportError as e:
        console.print(
            "\nPlease install all requirements for the full version of efemarai. Or run:\n"
            "python -m pip install efemarai\[full]",
            style="red",
        )
        raise e

    if hooks is None:
        hooks = []

    if input_format is None:
        input_format = ef.formats.DEFAULT_INPUT_FORMAT

    if class_ids is None:
        class_ids = _extract_class_ids(dataset, class_names, dataset_format)

    _log_version()

    class_weights = {class_id: 1 / len(class_ids) for class_id in class_ids}
    field_weights = defaultdict(lambda: 1)
    loss_fn = functools.partial(
        failure_loss,
        class_weights=class_weights,
        field_weights=field_weights,
    )

    search_args = _get_search_args(
        model=model,
        domain=domain,
        loss_fn=loss_fn,
        transform=transform,
        target_transform=target_transform,
        transforms=transforms,
        input_format=input_format,
        output_format=output_format,
        hooks=hooks,
        num_search_steps=num_search_steps,
        remap_class_ids=remap_class_ids,
    )

    with Progress(transient=True) as progress:
        task = progress.add_task("Testing", total=len(dataset))

        samples = pd.DataFrame()
        for index, (input, target) in enumerate(dataset):
            datapoint = ef.Datapoint.create_from(dataset_format, (input, target))

            if transform is not None:
                input = transform(input)

            if target_transform is not None:
                target = target_transform(target)

            if transforms is not None:
                input, target = transforms(input, target)

            output = model(input)

            output = ef.ModelOutput.create_from(
                output_format, datapoint, output, remap_class_ids=remap_class_ids
            )

            baseline_loss = loss_fn(datapoint, output)
            baseline_score = aggregate_loss(baseline_loss)

            search_args["baseline_datapoint"] = datapoint
            search_args["baseline_loss"] = baseline_loss
            search_args["baseline_score"] = baseline_score

            new_samples = _execute_search(search_args)
            if new_samples is not None:
                new_samples.assign(
                    image=index
                    if dataset_index_to_id is None
                    else dataset_index_to_id[index]
                )
                samples = pd.concat([samples, new_samples], ignore_index=True)

            progress.update(task, advance=1)

    report = ef.reports.RobustnessTestReport(samples)
    report.print_vulnerability()

    return report


def _extract_class_ids(dataset, class_names, dataset_format):
    if class_names is not None:
        return list(range(len(class_names)))

    class_ids = set()
    class_names = set()
    for _, target in track(
        dataset,
        description="Extracting classes info",
        transient=True,
    ):
        targets = ef.Datapoint.create_targets_from(dataset_format[1], target)
        labels = [target.label for target in targets if hasattr(target, "label")]
        class_ids.update([label.id for label in labels if label.id is not None])
        class_names.update([label.name for label in labels if label.name is not None])

    return class_ids or list(range(len(class_names)))


def _get_search_args(
    model,
    domain,
    loss_fn,
    transform,
    target_transform,
    transforms,
    input_format,
    output_format,
    hooks,
    num_search_steps,
    remap_class_ids=None,
    baseline_datapoint=None,
    baseline_loss=None,
    baseline_score=None,
):
    return {
        "model": model,
        "domain": domain,
        "loss_fn": loss_fn,
        "transform": transform,
        "target_transform": target_transform,
        "transforms": transforms,
        "input_format": input_format,
        "output_format": output_format,
        "hooks": hooks,
        "num_search_steps": num_search_steps,
        "remap_class_ids": remap_class_ids,
        "baseline_datapoint": baseline_datapoint,
        "baseline_loss": baseline_loss,
        "baseline_score": baseline_score,
    }


def _execute_search(args):
    try:
        from hyperactive import Hyperactive
        from hyperactive.optimizers import RepulsingHillClimbingOptimizer

    except ImportError as e:
        console.print(
            "\nPlease install all requirements for the full version of efemarai. Or run:\n"
            "python -m pip install efemarai\[full]",
            style="red",
        )
        raise e
    hyper = Hyperactive(verbosity=False)
    hyper.add_search(
        objective_function=_vulnerability_objective,
        search_space=args["domain"].search_space,
        n_iter=args["num_search_steps"],
        optimizer=RepulsingHillClimbingOptimizer(),
        initialize={"random": 3},
        pass_through=args,
    )

    try:
        hyper.run()
    except Exception as e:
        print(f"Stress testing failed: {e}")
        return None

    return hyper.search_data(_vulnerability_objective).drop_duplicates()


def _vulnerability_objective(opt):
    from efemarai.metamorph.loss.loss_functions import aggregate_loss, subtract_losses

    args = opt.pass_through

    input_format = args["input_format"]
    output_format = args["output_format"]

    transform = args["transform"]
    target_transform = args["target_transform"]
    transforms = args["transforms"]

    model = args["model"]
    domain = opt.pass_through["domain"]
    loss_fn = args["loss_fn"]
    remap_class_ids = args["remap_class_ids"]
    baseline_datapoint = opt.pass_through["baseline_datapoint"]
    baseline_loss = args["baseline_loss"]

    try:
        datapoint = domain.transform(baseline_datapoint, params=opt.para_dict)
    except Exception as e:
        print(f"Transformation failed: {e}")
        return float("nan")

    input = ef.spec.convert(input_format, datapoint)

    if transform is not None:
        input = transform(input)

    if target_transform is not None:
        target = target_transform(target)

    if transforms is not None:
        input, target = transforms(input, target)

    output = model(input)
    output = ef.ModelOutput.create_from(
        output_format, datapoint, output, remap_class_ids=remap_class_ids
    )

    sample_loss = loss_fn(datapoint, output)

    vulnerability = aggregate_loss(subtract_losses(sample_loss, baseline_loss))

    for hook in args["hooks"]:
        hook(
            baseline_datapoint,
            datapoint,
            output,
            baseline_loss,
            sample_loss,
            vulnerability,
        )

    return vulnerability


def _log_version():
    try:
        id = uuid.UUID(int=uuid.getnode())
        version = ef.__version__
        session = ef.Session(token="", url=ef.Session.DEFAULT_URL)
        process = mp.Process(
            target=session._post,
            args=("/api/logSdkVersion",),
            kwargs={"json": {"id": str(id), "version": version}, "verbose": False},
        )
        process.start()
    except:
        pass
