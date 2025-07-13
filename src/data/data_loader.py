import json
import io


def load_human_annotation_data(test_file):
    """Load human annotation file.
    """
    with io.open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_human_annotation_for_sample(ex, binary_ca):
    def _binarize(context_type):
        return "1" if context_type == "2" else "0"
    
    statements = ex["statements"]
    ops = [
        "delete-misleading",
        "delete-substandard",
        "delete-redundant",
        "add-evidence",
        "add-refinement", 
        "add-credibility"
    ]

    sent_id2types, sent_id2edits, sent_id2ratings, sent_id2citations = {}, {}, {}, {}
    for sid, sent_ex in statements.items():
        sent_id2citations[sid] = sent_ex["citations"]

        context_types = sent_ex["types"]
        if binary_ca:
            context_types = [_binarize(t) for t in context_types]
        sent_id2types[sid] = context_types
        
        sent_id2ratings[sid] = sent_ex["ratings"]
        
        # build edits
        edits = []
        for edit_annot in sent_ex["edits"]:
            n_edits = [len(edit_annot.get(op, [])) for op in ops]
            n_delete = sum(n_edits[:3])  # first 3 are deletes
            n_keep = len(sent_ex["citations"]) - n_delete
            n_edits += [n_keep]
            edits.append(n_edits)
        sent_id2edits[sid] = edits
    
    return sent_id2types, sent_id2edits, sent_id2ratings, sent_id2citations


def load_response_output(file_path, skip_data_processing=False):
    """Load evaluation data from `file_path`.
    """
    with open(file_path) as f:
        data = json.load(f)
    
    if skip_data_processing:
        return data
    
    raw_data = data
    data = []
    for item in raw_data:
        new_item = {
            'id': item["sample_idx"],
            'output': item['prediction'],
            'question': item['query'],
            "docs": [
                {'title': psg.get("title", ""), 'text': psg["text"]} for psg in item['passages']
            ],
            "sent_info": [
                {
                    "clean_sent": sent_and_cite["sentence"],
                    "citations": sent_and_cite["citations"],
                    "raw_sent": sent_and_cite["concat"]
                }       
                for sent_and_cite in item["prediction_sentences_and_citations"]
            ]
        }
        data.append(new_item)
    
    return data