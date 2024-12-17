import json
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report


def read_json(data_file):
    
    with open(data_file, 'r') as data_file:
        data = json.load(data_file)
    
    return data


def match_entities(data):

    matched_data = {}

    doc_id = 0
    for doc in data:
        doc_id += 1
        matched_data[f"doc_{doc_id}"] = {"length": len(doc["tokens"])}
        
        entity_id = 0
        for entity in doc["entities"]:
            entity_id += 1
            matched_data[f"doc_{doc_id}"][f"entity_{entity_id}"] = {"text": " ".join(doc["tokens"][entity["start"]:entity["end"]]),
                                                                    "label": entity["type"],
                                                                    "start": entity["start"],
                                                                    "end": entity["end"]}

    return matched_data, doc_id


def get_label_seq(matched_entities, no_docs):
    
    labels = []

    for i in range(no_docs):
        labels.append([])

        last_end = 0
        for entity_id in matched_entities[f"doc_{i+1}"]:
            if entity_id != "length":
                start = matched_entities[f"doc_{i+1}"][entity_id]["start"]
                end = matched_entities[f"doc_{i+1}"][entity_id]["end"]
                label = matched_entities[f"doc_{i+1}"][entity_id]["label"]

                assert start >= last_end

                if start > last_end:
                    for _ in range(start-last_end):
                        labels[i].append('O')

                labels[i].append(f'B-{label}')
                for _ in range(end-start-1):
                    labels[i].append(f'I-{label}')

                last_end = end
        
        for _ in range(matched_entities[f"doc_{i+1}"]["length"]-last_end):
            labels[i].append('O')
            
    return labels


# clean_data_file = "./noisebench/conll03_clean_train.json"
# original_data_file = "./noisebench/conll03_noisy_original_train.json"
# retrieved_data_file = "./results/conll03_noisy_original_train_lr_sum_k10.json"

clean_data_file = "./test.json"
original_data_file = "./test.json"
retrieved_data_file = "./results/test_lr_cls_k10.json"


clean_data = read_json(clean_data_file)
original_data = read_json(original_data_file)
retrieved_data = read_json(retrieved_data_file)

clean_matched_entities, no_clean_docs = match_entities(clean_data)
original_matched_entities, no_original_docs = match_entities(original_data)
retrieved_matched_entities, no_retrieved_docs = match_entities(retrieved_data)

assert no_clean_docs == no_original_docs == no_retrieved_docs
no_docs = no_clean_docs

clean_labels = get_label_seq(clean_matched_entities, no_docs)
original_labels = get_label_seq(original_matched_entities, no_docs)
retrieved_labels = get_label_seq(retrieved_matched_entities, no_docs)

print(f1_score(clean_labels, retrieved_labels))
print(classification_report(clean_labels, retrieved_labels))


for i in range(no_docs):
    assert len(original_labels[i]) == len(retrieved_labels[i])
    length = len(original_labels[i])
    
    for j in range(length):
        if original_labels[i][j] != retrieved_labels[i][j]:
            print(original_data[i]["tokens"])
            print(f"label index: {j}")
            print(f"clean label: {original_labels[i][j]}")
            print(f"retrieved label: {retrieved_labels[i][j]}")
            print("\n")

print("Done")