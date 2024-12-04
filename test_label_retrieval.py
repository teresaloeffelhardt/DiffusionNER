import label_retrieval as lr

import pytest


data_file_in = "./test.json"


def test_match_entities(data_file_in):
    
    data = lr.read_json(data_file_in)
    matched_data, origins = lr.match_entities(data, lmbda=0.35)

    assert "doc_1" in matched_data
    assert "doc_29" in matched_data
    assert "doc_30" not in matched_data

    assert matched_data["doc_1"]["orig_id"] == "335"
    assert matched_data["doc_29"]["orig_id"] == "336"

    # vor allem die nicht so obvious ones!


