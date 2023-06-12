<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://uploads-ssl.webflow.com/610aa229ecea935cd9cfb47a/610acaca4091b72c3fd40cf7_efemarai_logo_light-p-500.png#gh-dark-mode-only" width="400">
    <img src="https://uploads-ssl.webflow.com/610aa229ecea935cd9cfb47a/645b809d1044746ee26f2783_efemarai-logo-dark.png#gh-light-mode-only" width="400"/>
  </picture>
  
<div>&nbsp;</div>
  <div align="center">
    <a href="https://efemarai.com">
      <b><font size="5">Efemarai website</font></b>  
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">Efemarai platform</font></b>
    <sup>
      <a href="https://ci.efemarai.com">
        <i><font size="4">Use now!</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/efemarai)](https://pypi.org/project/efemarai)
[![license](https://img.shields.io/github/license/efemarai/efemarai.svg)](https://github.com/efemarai/efemarai/blob/main/LICENSE)

An SDK for interacting with the Efemarai ML [testing platform](https://ci.efemarai.com). Make your R&D model production ready.

[📘Documentation](https://ci.efemarai.com/docs) |
[🛠️Installation](https://ci.efemarai.com/docs/tutorials/getting_started.html#getting-started) |
[👀Break YOLO](https://breakyolo.efemarai.com/) |
[🚀Join Community](https://discord.gg/zXsVgSuemB) |
[😞Reporting Issues](https://github.com/efemarai/efemarai/issues/new/choose)

</div>

## Introduction

Efemarai is the easiest to integrate open source platform for testing and validating Computer Vision ML models. It works with any framework and model (PyTorch, TensorFlow, Keras, sklearn, Detectron2, OpenMMLab, YOLO, etc) and in 5 function calls finds examples that break your model.

<details open>
<summary>Major features</summary>

- **🔮 Operational Domain**

  Finetune how the images should be transformed, such that they cover the variablity the model is exepected to see in the real world. 

- **👨‍💻 Support any Input and Output types**

  Not only do we support tasks such as **classification**, **object detection**, **instance segmentation**, **keypoints detection**, **regression**, but also any combination thereoff, with any type of input - single image, multi-image, video, text, or anything that combines those.

- **📈 High efficiency**

  Don't waste time on randomly augmenting data, with Efemarai you are narrowing down failure modes in your model that are informative and you can fix.

</details>

## Example Works

### Find issues with a COCO detector

Apply advanced transformations, copy, edit, delete any part of the image, reimagine how things should vary in the real world.

![Break a detector](https://uploads-ssl.webflow.com/610aa229ecea935cd9cfb47a/645b8a746fad9c66d237495a_smaller-giraffe.gif)

### Break a face detector

With the `FaceWorks` module, you can perform relighting, face rotation, skin re-toning, etc to find conditions that break your model.

![Break face recognition](https://media.licdn.com/dms/image/C4D22AQFgBefVcNSV7A/feedshare-shrink_800/0/1678370502298?e=1686787200&v=beta&t=VqXHpbQieUkdHesGO5KwHxVvAxGAyB_RrzGUtr-QCZU)

#### Other

If you work in the medical, aerospace and defence, security or ag domain, we provide custom capabilities that are domain specific. For these or other enquiries, drop us [a message](mailto:svet@efemarai.com?subject="Do%20you%20work%20with%20this?").

## Setup

Install full version with
```shell
pip install -U efemarai[full]
```
or just base with `pip install -U efemarai`; then run
```shell
efemarai init
```
and follow the instructions to connect your account to [https://ci.efemarai.com](https://ci.efemarai.com).

## Example Local Usage

### NLP/LLMs

Testing Natural Langugage and LLMs allow you to understand how sensitive is your model embedding to common human input errors like spelling mistakes, synonyms, or plain swapping of names and places. 

Have a look at the examples [here](https://ci.efemarai.com/docs/local/getting_started/bert_squad.html).

#### Test BERT locally

Here is a code example on how to get started testing a BERT model with the SQUAD dataset:

```
import efemarai as ef
from datasets import load_dataset, logging as datasets_logging
from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    logging as transformers_logging,
)

datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


def get_model_func(model, tokenizer):
    def predict(inputs):
        inputs = tokenizer(inputs["question"], inputs["context"], return_tensors="pt")
        outputs = model(**inputs)

        start_index = outputs.start_logits.argmax()
        end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, start_index : end_index + 1]
        result = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

        return result

    return predict


def main():
    dataset = load_dataset("squad", split="validation[:10]")
    dataset = [
        (
            {"context": dp["context"], "question": dp["question"]},
            {"answer": dp["answers"]["text"][0]},
        )
        for dp in dataset
    ]

    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

    report = ef.test_robustness(
        dataset=dataset,
        model=get_model_func(model, tokenizer),
        domain=ef.domains.TextVariability,
        dataset_format=ef.formats.TEXT_EQA_DATASET,
        output_format=lambda text: ef.Text(text=text),
        input_format=lambda datapoint: {
            "context": datapoint.context.text,
            "question": datapoint.question.text,
        },
    )

    report.plot("robustness_report.pdf")


if __name__ == "__main__":
    main()
```    

### Computer Vision 

We support a large range of integrations with different open source models, as well as a set of pre-defined domains (`ef.domains.GeometricVariability`, `ef.domains.ColorVariability`, `ef.domains.NoiseVariability`) to kickstart your ML testing. 

Here you can find a few tutorials integrating a [`torchvision` model with COCO dataset](https://ci.efemarai.com/docs/local/getting_started/torchvision_coco.html), a [`Detectron2 Mask-RCNN` model with a balloon dataset](https://ci.efemarai.com/docs/local/getting_started/detectron2_instance_balloons.html), a [`YOLO` model](https://ci.efemarai.com/docs/local/getting_started/yolo_ultralytics_coco.html) or a [Roboflow model](https://ci.efemarai.com/docs/local/getting_started/yolo_roboflow_coladetect.html).

The included code examples can start you from zero to one and get insight into the model performance like:

```shell
    Robustness Test Report    
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃    Axis    ┃ Vulnerability ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ saturation │    0.2825     │
│ brightness │    0.2718     │
│  contrast  │    0.2027     │
│   value    │    0.2022     │
│   gamma    │    0.2014     │
│    hue     │    0.1130     │
└────────────┴───────────────┘
✔ Report plot saved as 'robustness_report.pdf'
```

## Example Hub Usage

### Create a Bounding Box Project
When your project depends on bounding boxes, the uploaded dataset needs to contain the required bounding box information alongside each image as part of a single datapoint.

First we will create a dataset, and later on, a dummy model that returns bounding box information.

#### Create dataset
A convenient approach for creating a bounding box dataset is by ensuring that the local format is in COCO-compatible format.

```python
import efemarai as ef

# Create a project
project = ef.Session().create_project(
    name="Example Bounding Box Project (COCO)",
    description="Example project using the COCO dataset format.",
    exists_ok=True,
)

dataset = project.create_dataset(
    name="Bounding Box dataset",
    data_url="./data/coco/test",
    annotations_url="./data/coco/annotations/test_instances.json",
    stage=ef.DatasetStage.Test,
    format=ef.DatasetFormat.COCO,
)
```
If your dataset is remote or part of an existing database with custom formats, you can easily upload it to the system by (1) iterating over the dataset and (2) creating datapoints containing the images and required targets. You can find a code example [here](https://ci.efemarai.com/docs/tutorials/how_to/project_bounding_box.html#create-dataset).

After wrapping up any processing, you can confirm the status in the UI and explore the inputs and annotations.

#### Create a model
A model that works with bounding boxes dataset will need to return a list of `ef.BoundingBox` objects that will be matched to the ones stored in the dataset. In a file `dummy_model.py` save the following code:

```python
import efemarai as ef
import numpy as np

class DummyModel:
    """A DummyModel returning a random bbox"""

    def __init__(self, device):
        self.device = device # Move model to device

    def __call__(self, image):
        return {
            "class_id": np.random.randint(0, 3),
            "bbox": [100, 150, 250, 350],
            "score": np.random.random(),
        }


def predict_images(datapoints, model, device):
    outputs = []
    for datapoint in datapoints:
        image = datapoint.get_input("image") # This corresponds to the key from the datapoint input creation dict

        image_post_process = image.data / 255 - 0.5 # perform any pre-processing

        output = model(image_post_process)

        # Here again the label can be referenced by name or class
        # label = ef.AnnotationClass(name=output["class_name"])
        label = ef.AnnotationClass(id=output["class_id"])

        outputs.append(
            [
                ef.BoundingBox(
                    xyxy=output["bbox"],
                    confidence=output["score"], # Confidence of detection
                    ref_field=image, # Say which image this output refers to
                    label=label,     # And what label it has
                ),
            ]
        )
    return outputs


def load_model(device):
    model = DummyModel(device)
    return model
```

That's the two things you need to define - how to load the model and how to perform an inference on a batch!

#### efemarai.yaml file
To run the model, you need to have defined the loading and inference capabilities in the efemarai.yaml file. This way we can work with any model from any platfrom or framework.

Here’s the one corresponding to the dummy model.

```yaml
project:
  name: "Example Bounding Box Project"

models:
  - name: Dummy Model
    description: This is a dummy model to show consuming inputs and outputs

    runtime:
      image: python:3.10-slim-buster
      device: "gpu"
      batch:
        max_size: 10
      load:
        entrypoint: dummy_model:load_model
        inputs:
          - name: device
            value: ${model.runtime.device}
        output:
          name: model

      predict:
        entrypoint: dummy_model:predict_images
        inputs:
          - name: datapoints
            value: ${datapoints}
          - name: model
            value: ${model.runtime.load.output.model}
          - name: device
            value: ${model.runtime.device}
        output:
          name: predictions
          keys:
            - bbox
```

#### Register the model
To register the model, use the CLI to upload it by going into the root of the file directory, next to the `efemarai.yaml`.

```shell
ef model create .
```
Now you should be able to see the model uploaded and active with this project.

### Create a domain

Let's use the UI to quickly create a domain (`Example`) that you expect your model to operate in.

![](https://storage.googleapis.com/public-efemarai/domain2.gif)

You can find more information in the [docs](https://ci.efemarai.com/docs/tutorials/step_by_step/domain.html#create-domain).

### Working with stress tests

Now that you have defined your model, domain and dataset, you are ready to stress test your model and validate how well it works!

```python
# Create a new stress test
test = project.create_stress_test(
    name="Test via SDK",
    model=project.model("Dummy Model"),
    domain=project.domain("Example"),
    dataset="Example Bounding Box Project (COCO)", # Just a name also works
)

# Load an existing stress test
test = project.stress_test("Test via SDK")

# Download dataset with discovered vulnerabilities
dataset_filepath = test.vulnerabilities_dataset()

# Check test run state
print(f"Running: {test.running} Failed: {test.failed} Finished: {test.finished}")
```

Models, domains and datasets can be easily created programatically, but
they require quite a few configuration paramaters to be provided. That's
why the most convenient way to create a project with multiple models, domains
and datasets is to put everything into a config file.
