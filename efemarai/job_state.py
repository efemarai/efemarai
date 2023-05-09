from enum import Enum


class JobState(Enum):
    """Possible Test Run, Baseline and Dataset States."""

    NotStarted = "NotStarted"
    Starting = "Starting"
    Evaluating = "Evaluating"
    Testing = "Testing"
    Loading = "Loading"
    DataLoaded = "Data Loaded"
    ExtractingAssets = "Extracting Assets"
    CalculatingMetadata = "Calculating Metadata"
    Enhancing = "Enhancing"
    GeneratingReport = "Generating Report"
    Finished = "Finished"
    Failed = "Failed"
    Stopped = "Stopped"
