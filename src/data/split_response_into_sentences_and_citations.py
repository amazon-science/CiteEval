import re
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize


def build_cite_core_regex(left_cite_token="[", right_cite_token="]", allow_multi_numbers_in_brackets=False):
    """Return the core regex for recoganizing citations.
    """
    left = re.escape(rf'{left_cite_token}')
    right = re.escape(rf'{right_cite_token}')

    
    if allow_multi_numbers_in_brackets:
        core = "(\d+(,(\s)?\d+)*)"
    else:
        core = "(\d+)"

    core = rf"{left}{core}{right}"
    return core


def get_citations(sent, core_cite_regex):
    """Return a list of (int) citations extracted from `sent`.
    `core_cite_regex`: the core regex for recoganizing citations. Example: r"\[(\d+)\]".
    """
    matches = re.findall(core_cite_regex, sent)
    if len(matches) == 0:
        return []
    
    citations = []
    for match in matches:
        if isinstance(match, tuple):
            for citation in match[0].split(","):  # the first group matches all bracketed info, such as 1,2,3
                citations.append(int(citation.strip()))
        else:
            citations.append(int(match))
    
    return citations


def remove_citations(sent, core_cite_regex):
    """Remove citations from `sent` and return a clean sentence. 
    `core_cite_regex`: the core regex for recoganizing citations. Example: r"\[(\d+)\]".
    """
    regex = rf"\s*{core_cite_regex}"  # example: r"\s*\[\d+\]"
    return re.sub(regex, "", sent)


def check_prefix_citations(sent, core_cite_regex):
    """Check if there are prefix citations at the beginning of `sent`. 
    `core_cite_regex`: the core regex for recoganizing citations. Example: r"\[(\d+)\]".
    Return the end token index of the prefix, or -1 if the prefix does not exist.
    """
    regex = rf"^(?:\W*)(({core_cite_regex})+)"
    match = re.match(regex, sent)
    if match:
        _, end = match.span()
        return end
    return -1



no_answer_prediction_regex = re.compile(
    r'(?i)\b(no|none|not)\b.{0,30}(\binformation\b|\bdetails\b|\banswer\b|\bmention(ed)?\b|\bprovided\b|\bsources?\b|\bpassages?\b|\bfound)\b'
)


def is_no_answer_prediction(prediction: str) -> bool:
    """ Check whether an LLM prediction is likely a "no answer" prediction. This checking is currently done with a regex pattern
    curated from Claude-instant responses. As a result, it may only have good coverage for Claude models.
    """
    if re.search(no_answer_prediction_regex, prediction):
        return True
    else:
        return False


def build_sent_info(raw_sent, core_cite_regex, detect_prefix_citations):
    """Build sentence information dict for a given sentence, consisting of:
        `raw_sent`: 
            the input sentence
        `core_cite_regex`: 
            the core regex for recoganizing citations. Example: r"\[(\d+)\]".
        `citations`: 
            citations extracted for the given sentence
        `detect_prefix_citations`:
            detect citations extracted at the sentence begining, which should be part of the previous sentence, if applicable.
    """
    clean_sent = remove_citations(raw_sent, core_cite_regex).strip()
    
    prefix_citations = None
    seq_for_curr_citations = raw_sent
    if detect_prefix_citations:  # detect citations placed at sentence begining
        end = check_prefix_citations(raw_sent, core_cite_regex)
        if end == -1:
            prefix_citations = []
        else:
            prefix_citations = get_citations(raw_sent[:end], core_cite_regex)
            seq_for_curr_citations = raw_sent[end:]
    
    citations = get_citations(seq_for_curr_citations, core_cite_regex)

    sent_info = {
        'raw_sent': raw_sent,
        'clean_sent': clean_sent,
        'citations': citations,
    }

    if detect_prefix_citations:
        sent_info['prefix_citations'] = prefix_citations
    
    return sent_info


def build_sent_info_for_a_response(
        response, 
        core_cite_regex, 
        postproc_citations, 
        n_passages=10, 
        max_n_citations=3, 
        merge_short_sentences=True, 
        add_one_to_citation=False,
        citation_map=None
    ):
    """Build sentence-level info for a given response. 
    `core_cite_regex`: the core regex for recoganizing citations. Example: r"\[(\d+)\]".
    Return a list of `sent_info` dicts.

    citation_map: map citation sentence id to passage id (only applicable to lqac).
    """
    response = response.replace("\n\n", "\n")
    segments = response.split("\n")
    
    sents = []
    for seg_id, seg in enumerate(segments):
        _sents = sent_tokenize(seg)
        if seg_id < len(segments)-1:
            _sents[-1] += '\n'
        sents.extend(_sents)  # get sentences via NLTK
    
    sent_info = []
    for sent in sents:
        curr = build_sent_info(sent, core_cite_regex, detect_prefix_citations=True)
        
        # handle prefix_citations: re-attached it to prev citations if applicable
        if 'prefix_citations' in curr and curr['prefix_citations']:
            if sent_info:
                sent_info[-1]['citations'].extend(curr['prefix_citations'])
            else:
                curr['citations'].extend(curr['prefix_citations'])
        
        if curr['clean_sent']:  # skip empty sentences
            sent_info.append(curr)
    
    
    # merge extremely short sentences under `short_sentence_chars`
    # the target sentence has citations at its end, or have \n at its end
    # for these cases we know the sentence should not be merged into its subsequent sentence
    if merge_short_sentences:
        short_sentence_chars = 20
        new_sent_info = []
        for sinfo in sent_info:
            if len(new_sent_info)==0 or len(sinfo['clean_sent']) >= short_sentence_chars:
                # print(f"Add {sinfo['clean_sent']}. Length: {len(sinfo['clean_sent'])}")
                new_sent_info.append(sinfo)
                continue

            if sinfo['raw_sent'].endswith("\n") or len(sinfo['citations'])>0:
                new_sent_info[-1]['clean_sent'] += " " + sinfo['clean_sent']
                new_sent_info[-1]['citations'].extend(sinfo['citations'])
            else:
                new_sent_info.append(sinfo)

        sent_info = new_sent_info
    
    if postproc_citations:
        for sinfo in sent_info:
            citations = [citation for citation in set(sinfo['citations']) if 1 <= citation <= n_passages]
            
            if max_n_citations is not None:
                citations = citations[:max_n_citations]
            
            sinfo['citations'] = citations
    
    
    for sinfo in sent_info:
        citations = sinfo['citations']

        if add_one_to_citation:  # when the model citation starts at 0 -- the sources will start at 1 for eval
            citations = [citation+1 for citation in citations]
        
        if citation_map is not None:
            mapped_citations = []
            for citation in citations:
                if citation_map[citation] in mapped_citations:
                    continue
                mapped_citations.append(citation_map[citation])
            citations = mapped_citations
        
        sinfo['citations'] = citations

    return sent_info


def postprocess_prediction_by_removing_thinking(prediction, open_bracket, close_bracket):
    """ We post-process a model response by removing the thinking process from it. This is useful when we
    ask the LLM to think step by step but we only want to evaluate on the actual answer without the thinking process.

    We assume that the thinking process comes as a continuous chunk at the beginning of the response.

    Returns:
        prediction: the post-processed response string
        thinking_found: whether a thinking process is found and removed in response
        thinking_length: word length of the thinking process, without brackets.
    """
    prediction = prediction.strip()
    thinking_start_idx = prediction.find(open_bracket)
    thinking_close_idx = prediction.find(close_bracket)

    thinking_found = False
    thinking_length = 0
    if thinking_start_idx >= 0 and thinking_close_idx >= 0 and thinking_start_idx <= thinking_close_idx:
        # we found both start and close brackets and they are valid
        thinking_found = True
        # we extract the thinking substring out
        thinking_str = prediction[thinking_start_idx+len(open_bracket): thinking_close_idx]
        thinking_length = len(word_tokenize(thinking_str))
    # otherwise we miss at least one bracket

    # we postprocess the prediction by removing anything before the thinking close bracket (including bracket)
    if thinking_close_idx >= 0:
        prediction = prediction[thinking_close_idx+len(close_bracket):].strip()

    return prediction, thinking_found, thinking_length


def get_sentences_and_citations(
        response, 
        max_n_citations=None, 
        return_sent_info=False, 
        replace_incomplete_with_abstention=False, 
        is_lqac=False,
        citation_map=None
    ):
    if is_lqac:  # lqac-specific post-processing logic
        tags_to_remove = ["<statement>", "</statement>", "<cite>", "</cite>"]
        for tag in tags_to_remove:
            response = response.replace(tag, "")
        postproc_citations = False  # lqac can have >10 passages; do not remove citations > 10

        if citation_map:
            add_one_to_citation = False  # do not add one; map sid => pid directlt
        else:    
            add_one_to_citation = True  # lqac citations start from 0: we should +1
    else:
        postproc_citations = True
        add_one_to_citation = False
    
    response, _, _ = postprocess_prediction_by_removing_thinking(prediction=response, open_bracket="<thinking>", close_bracket="</thinking>")

    if replace_incomplete_with_abstention and response.strip().startswith("<thinking>"):
        print(f"Replacing incomplete response with abstention:\nRaw response: {response}\n")
        response = "No answer is found."

    if is_no_answer_prediction(response.strip()):
        response = "No answer is found."
    
    def _format(sentence, citations):
        if not citations:
            return sentence
        return sentence + ' ' + ''.join([f"[{cite}]" for cite in citations])
    
    core_cite_regex = build_cite_core_regex(left_cite_token="[", right_cite_token="]", allow_multi_numbers_in_brackets=False)
    sent_info = build_sent_info_for_a_response(
        response, 
        core_cite_regex, 
        postproc_citations=postproc_citations, 
        max_n_citations=max_n_citations, 
        merge_short_sentences=True, 
        add_one_to_citation=add_one_to_citation,
        citation_map=citation_map
    )

    if return_sent_info:
        return sent_info
    
    sentences_and_citations = [(sinfo["clean_sent"], sinfo["citations"], _format(sinfo["clean_sent"], sinfo["citations"])) for sinfo in sent_info]
    return response, sentences_and_citations
