import efemarai as ef

from torchvision import datasets
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


def main():
    # To download the COCO dataset you could use the following steps:
    #
    # mkdir -p data/coco
    # cd data/coco; \
    # 	wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip; \
    # 	wget -c http://images.cocodataset.org/zips/val2017.zip; \
    # 	unzip annotations_trainval2017.zip; \
    # 	unzip val2017.zip;

    path = "data/coco"

    dataset = datasets.CocoDetection(
        root=f"{path}/val2017",
        annFile=f"{path}/annotations/instances_val2017.json",
    )
    dataset = [dataset[i] for i in range(5)]

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.cuda()
    model.eval()

    report = ef.test_robustness(
        dataset=dataset,
        model=lambda x: model(x.cuda().unsqueeze(0))[0],
        domain=ef.domains.ColorVariability,
        dataset_format=ef.formats.COCO_DATASET,
        output_format=ef.formats.TORCHVISION_DETECTION,
        transform=weights.transforms(),
        class_ids=range(100),
        hooks=[ef.hooks.show_sample],
    )

    report.plot("robustness_report.pdf")


if __name__ == "__main__":
    main()
