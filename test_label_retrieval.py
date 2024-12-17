import label_retrieval as lr

import pytest


@pytest.fixture
def data():
    data_file_in = "./test.json"
    data = lr.read_json(data_file_in)
    return data

@pytest.fixture
def match_35(data):
    matched_data, origins = lr.match_entities(data, lmbda=0.35)
    return matched_data, origins

@pytest.fixture
def match_0(data):
    matched_data, origins = lr.match_entities(data, lmbda=0)
    return matched_data, origins


def test_match_entities(match_35):
    
    matched_data, origins = match_35

    assert "doc_1" in matched_data
    assert "doc_29" in matched_data
    assert "doc_30" not in matched_data

    assert matched_data["doc_1"]["orig_id"] == "335"
    assert matched_data["doc_29"]["orig_id"] == "336"

    assert matched_data["doc_1"]["entity_1"] == {"text": "German",
                                                 "label": "MISC",
                                                 "start": 0,
                                                 "end": 1}
    assert matched_data["doc_3"]["entity_2"] == {"text": "Agriculture Ministry",
                                                 "label": "ORG",
                                                 "start": 2,
                                                 "end": 4}
    

    assert origins["335"]["docs"]["doc_3"] == ["Germany",
            "'s",
            "Agriculture",
            "Ministry",
            "suggested",
            "on",
            "Wednesday",
            "that",
            "consumers",
            "avoid",
            "eating",
            "meat",
            "from",
            "British",
            "sheep",
            "until",
            "scientists",
            "determine",
            "whether",
            "mad",
            "cow",
            "disease",
            "can",
            "be",
            "transmitted",
            "to",
            "the",
            "animals",
            "."]
    assert origins["335"]["n_entities"] > 32


def test_negative_sampling_doc(match_35, match_0):

    matched_data35, origins35 = match_35

    o = False
    neg_entity = ""
    for entity_id in matched_data35["doc_3"]:
        if entity_id != "orig_id":
            if matched_data35["doc_3"][entity_id]["label"] == "O":
                o = True
                neg_entity = matched_data35["doc_3"][entity_id]["text"]
    assert o == True
    assert len(neg_entity) > 0
    assert origins35["335"]["n_entities"] > 32

    matched_data0, origins0 = match_0

    o = True
    for entity_id in matched_data0["doc_3"]:
        if entity_id != "orig_id":
            if matched_data0["doc_3"][entity_id]["label"] == "O":
                o = False     
    assert o == True
    assert origins0["335"]["n_entities"] == 32


def test_similarity_embedding_cls(match_0):

    matched_data, origins = match_0
    lr.similarity_embedding_cls(matched_data, origins)

    assert matched_data["doc_3"]["entity_2"]["embedding"].shape == (1, 1024)
    assert matched_data["doc_3"]["entity_2"]["index"] == 4
    assert origins["335"]["embeddings"].shape == (origins["335"]["n_entities"], 1, 1024)


def test_similarity_embedding_sum(match_0):

    matched_data, origins = match_0
    lr.similarity_embedding_sum(matched_data, origins)

    assert matched_data["doc_3"]["entity_2"]["embedding"].shape == (1, 1024)
    assert matched_data["doc_3"]["entity_2"]["index"] == 4
    assert origins["335"]["embeddings"].shape == (origins["335"]["n_entities"], 1, 1024)