import io
import json
import argparse
from data.split_response_into_sentences_and_citations import get_sentences_and_citations


def postproc_model_response_sample(sample):
    """Post process `sample`, a model prediction sample: 

    - Format passages 
    - Split response into sentences
    - Extract and postprocess citation
    - Add IDs for sentences, passages, and the sample
    """
    response = sample["pred"]
    passages = sample["passages"]

    postproc_pred, sentences_and_citations = get_sentences_and_citations(response)
    
    sentence_objs = []
    sent_id = 1
    for sent, citations, concatenated in sentences_and_citations:
        sentence_obj = {
            "id": str(sent_id), 
            "sentence": sent,
            "citations": citations,
            "concat": concatenated
        }
        sentence_objs.append(sentence_obj)
        sent_id += 1
    
    new_passages = []
    for psg_idx, psg in enumerate(passages):
        new_psg = {
            "id": str(psg_idx + 1),
            "title": psg.get("title", ""),
            "text": psg["text"]
        }
        new_passages.append(new_psg)
    
    new_obj = {
        "sample_idx": sample["id"],
        "query": sample["query"],
        "passages": new_passages,
        "prediction": postproc_pred,
        "prediction_sentences_and_citations": sentence_objs
    }

    return new_obj


def convert(system_output_file):
    formatted_data = []

    system_output_fn = system_output_file.split("/")[-1]
    system_output_dir = "/".join(system_output_file.split("/")[:-1])

    formatted_system_output_fn = system_output_fn.replace(".json", ".citeeval")
    formatted_system_output_file = f"{system_output_dir}/{formatted_system_output_fn}"

    with io.open(system_output_file) as pred_f:
        preds = json.load(pred_f)
        for pred in preds:
            postproc_pred = postproc_model_response_sample(pred)
            formatted_data.append(postproc_pred)

    with io.open(formatted_system_output_file, 'w') as out_f:
        json.dump(formatted_data, out_f, indent=2)

    print(f"Finish: {system_output_file} => {formatted_system_output_file}. Data size: {len(formatted_data)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_output_file", type=str, required=True, help="System output prediction file.")
    args = parser.parse_args()

    convert(system_output_file=args.system_output_file)
