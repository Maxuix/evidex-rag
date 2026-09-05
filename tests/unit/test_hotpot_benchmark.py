"""Verify portable Hotpot evidence offsets and corpus isolation, offline."""
import pytest

from tools.build_hotpot_benchmark import derive


def test_unicode_evidence_offsets_and_distractors_are_preserved():
    original = [{"_id":"q1", "question":"Question kept outside corpus?",
                 "answer":"Gold kept outside corpus", "type":"bridge", "level":"hard",
                 "context":[["中文 title",["First. ","第二句。"]], ["Noise",["Distractor."]]],
                 "supporting_facts":[["中文 title",1]]}]
    questions, cases, docs = derive(original)
    assert questions == [{"case_id":"q1","question":"Question kept outside corpus?"}]
    assert len(docs) == 2
    assert {d["title"] for d in docs} == {"中文 title","Noise"}
    evidence = cases[0]["supporting_facts"][0]
    assert evidence["char_start"] == 17
    assert evidence["char_end"] == 21
    assert evidence["quote"] == "第二句。"
    assert evidence["sentence_index"] == 1
    assert all("Gold kept outside corpus" not in d["text"] for d in docs)
    assert all("Question kept outside corpus?" not in d["text"] for d in docs)


def test_invalid_sentence_reference_is_rejected():
    row = {"_id":"q", "question":"Q", "answer":"A", "type":"bridge", "level":"hard",
           "context":[["Title",["Sentence."]]], "supporting_facts":[["Title",902]]}
    with pytest.raises(ValueError, match="out of bounds"):
        derive([row])
