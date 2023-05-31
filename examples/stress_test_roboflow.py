import efemarai as ef
import supervision as sv
from roboflow import Roboflow
import os


def main():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("projeeeeeeeeeeeeeee").project("coladetect").version(13)
    dataset = project.download("yolov8")
    model = project.model

    dataset_iter = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/test/images",
        annotations_directory_path=f"{dataset.location}/test/labels",
        data_yaml_path=f"{dataset.location}/data.yaml",
    )
    dataset_classes = dict(
        zip(dataset_iter.classes, list(range(len(dataset_iter.classes))))
    )

    dataset = []
    for _, img, detections in dataset_iter:
        data = (img[:, :, ::-1], detections)
        dataset.append(data)

    def model_func(x):
        output = model.predict(x, confidence=40, overlap=30).json()
        for pred in output["predictions"]:
            pred["class"] = dataset_classes[pred["class"]]
        return output

    report = ef.test_robustness(
        dataset=dataset,
        model=model_func,
        domain=ef.domains.ColorVariability,
        dataset_format=ef.formats.SUPERVISION_DETECTION,
        input_format=ef.formats.DEFAULT_INPUT_NP_FORMAT,
        output_format=ef.formats.ROBOFLOW_DETECTION,
        class_ids=list(range(len(dataset_iter.classes))),
        hooks=[ef.hooks.show_sample],
    )

    report.plot("robustness_report.pdf")


if __name__ == "__main__":
    main()
