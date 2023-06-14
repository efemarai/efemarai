import efemarai as ef
import numpy as np
import os, json, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            # Expected Efemarai Polygon format: List[List[List[floats]]]
            poly = [[x + 0.5, y + 0.5] for x, y in zip(px, py)]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_model():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.15
    return DefaultPredictor(cfg)


def main():
    # Make sure you have downloaded the dataset
    # !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    # !unzip balloon_dataset.zip > /dev/null

    dataset_dict = get_balloon_dicts("balloon/val")
    model = get_model()

    dataset = []
    for record in dataset_dict:
        dataset.append(
            (cv2.imread(record["file_name"])[:, :, ::-1], record["annotations"])
        )

    report = ef.test_robustness(
        dataset=dataset,
        model=lambda x: model(x)["instances"],
        domain=ef.domains.ColorVariability,
        dataset_format=ef.formats.COCO_DATASET,
        input_format=ef.formats.DEFAULT_INPUT_NP_FORMAT,
        output_format=ef.formats.DETECTRON_OBJECT_DETECTION,
        class_ids=list(range(100)),
        hooks=[ef.hooks.show_sample],
    )

    report.plot("robustness_report.pdf")


if __name__ == "__main__":
    main()
