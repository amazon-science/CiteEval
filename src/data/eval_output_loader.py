import io
import json


def load_autoais_ratings(test_file):
    with open(test_file) as f:
        examples = json.load(f)
    
    sample_id2ratings_model = {}
    for ex in examples:
        sample_id = ex["id"]
        sample_id2ratings_model[sample_id] = ex
    
    return sample_id2ratings_model


def load_attriscore_ratings(test_file):
    """Load AttriScore ratings from its prediction file.
    """
    def aggregate(sent_preds, pred2rating_map, mode, default_rating=1.0):
        """Take the pred for each citation, and aggregate them into a single statement-level score.
        """
        assert mode in ["strict", "relaxed"]

        citation_ratings = []
        for pred in sent_preds:
            if pred == "Extrapolatory":
                pred = f"{pred}.{mode}"
            
            rating = pred2rating_map.get(pred, default_rating)
            citation_ratings.append(rating)
        
        sent_rating = sum(citation_ratings) / len(citation_ratings)
        return sent_rating

    
    with open(test_file) as f:
        items = json.load(f)

    pred2rating_map = {
        "Attributable": 1.0,
        "Extrapolatory.strict": 0.0,
        "Extrapolatory.relaxed": 0.5,
        "Contradictory": 0.0,
    }

    sample_id2ratings_model = {}

    for item in items:
        sid2rating_strict = {}
        sid2rating_relaxed = {}
        
        sample_id = None
        
        for sent_examples in item:
            sent_id = None
            
            sent_preds = []
            for ex in sent_examples:
                if ex["citation"] == "":
                    pred = "Contradictory"
                else:
                    pred = ex["pred"]
                sent_preds.append(pred)

                if sample_id is None:
                    sample_id = ex["id"]
                else:
                    assert sample_id == ex["id"]

                if sent_id is None:
                    sent_id = ex["sent_id"]
                else:
                    assert sent_id == ex["sent_id"]
                
            sent_rating_strict = aggregate(sent_preds, pred2rating_map=pred2rating_map, mode="strict")
            sent_rating_relaxed = aggregate(sent_preds, pred2rating_map=pred2rating_map, mode="relaxed")
            
            sid2rating_strict[sent_id] = sent_rating_strict
            sid2rating_relaxed[sent_id] = sent_rating_relaxed
        
        sample_id2ratings_model[sample_id] = {
            "sid2rating_strict": sid2rating_strict,
            "sid2rating_relaxed": sid2rating_relaxed,
        }
    
    return sample_id2ratings_model



def load_citeval_ratings_and_types(test_file, binary_ca, config, sample_id2sent_citations):
    def _binarize(ca_pred, sent_citations):
        yes_class = None
        for k, v in config["context_types2citation_requirement"].items():
            if v == "yes":
                yes_class = k
                break
        if not yes_class:
            raise ValueError(f"Cannot find yes class in config: {config}")
        
        if sent_citations:  # has citations => citations required
            ca_pred = yes_class
        
        if binary_ca:
            return "1" if ca_pred == yes_class else "0"
        
        return ca_pred

    with io.open(test_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    sample_id2sent_ratings = {}
    sample_id2sent_types = {}

    for ex in preds:
        assert ex["id"] in sample_id2sent_citations

        sent_citations = sample_id2sent_citations[ex["id"]]

        sample_id2sent_ratings[ex["id"]] = ex["sent_id2rating"]
        sample_id2sent_types[ex["id"]] = {}

        for sid, value in ex["sent_id2type"].items():
            assert sid in sent_citations, f'sample id: {ex["id"]}, sid: {sid} not found in sent_citations: {sent_citations}'
            sent_type = _binarize(value["ca_pred"], sent_citations[sid])
            sample_id2sent_types[ex["id"]][sid] = sent_type
    
    return sample_id2sent_ratings, sample_id2sent_types
