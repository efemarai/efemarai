import efemarai as ef
from transformers import logging as transformers_logging
from transformers import ViltProcessor, ViltForQuestionAnswering

import json
import numpy as np
from PIL import Image

transformers_logging.set_verbosity_error()


def read_coco_vqa(coco_path):
    with open(f"{coco_path}/annotations/vqa_answers.json") as answers_rfile, open(
        f"{coco_path}/annotations/vqa_questions.json"
    ) as questions_rfile, open(
        f"{coco_path}/annotations/instances_val2017.json"
    ) as images_rfile:
        answers = json.load(answers_rfile)["annotations"]
        questions = json.load(questions_rfile)["questions"]
        images = json.load(images_rfile)["images"]
    return answers, questions, images


def prepare_coco_vqa():
    coco_path = "data/coco"

    answers, questions, images = read_coco_vqa(coco_path)
    images = {image["id"]: image["file_name"] for image in images}
    images_in_dataset = []
    dataset = []
    for question, answer in zip(questions[:10], answers[:10]):
        image_name = images.get(answer["image_id"])
        if image_name is not None and image_name not in images_in_dataset:
            image = np.array(Image.open(f"{coco_path}/val2017/{image_name}"))
            dataset.append(
                (
                    {"image": image, "question": question["question"]},
                    {"answers": answer["answers"][0]["answer"]},
                )
            )
            images_in_dataset.append(image_name)
    return dataset


def main():
    dataset = prepare_coco_vqa()

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def model_func(x):
        # prepare inputs
        encoding = processor(x["image"], x["question"], return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return model.config.id2label[idx]

    report = ef.test_robustness(
        dataset=dataset,
        model=model_func,
        domain=ef.domains.TextVariability,  # ef.domains.TextGPTVariability
        dataset_format=ef.formats.VQA_DATASET,
        output_format=lambda text: ef.Text(text=text),
        input_format=lambda datapoint: {
            "image": datapoint.image.data,
            "question": datapoint.question.text,
        },
        class_ids=[],
    )

    report.plot("robustness_report.pdf")


if __name__ == "__main__":
    main()
