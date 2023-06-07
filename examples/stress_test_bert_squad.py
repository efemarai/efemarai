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
        num_search_steps=10,
    )

    report.plot("robustness_report.pdf")


if __name__ == "__main__":
    main()
