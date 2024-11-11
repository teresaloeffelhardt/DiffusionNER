import json
import math
import random
from transformers import BertConfig, BertTokenizer
from diffusionner.modeling_bert import BertEmbeddings as DiffusionNERBertEmbeddings
import torch
from collections import Counter

import pprint


def read_json(data_file):

    # vergabe von orig_id überprüfen
    
    with open(data_file, 'r') as data_file:
        data = json.load(data_file)
    
    return data


def match_entities(data, lmbda):

    matched_data = {}

    doc_id = 0
    for doc in data:
        doc_id += 1
        matched_data[f"doc_{doc_id}"] = {}
        
        pos_entities = []
        neg_entities = []

        entity_id = 0
        for entity in doc["entities"]:
            entity_id += 1
            matched_data[f"doc_{doc_id}"][f"entity_{entity_id}"] = {"text": " ".join(doc["tokens"][entity["start"]:entity["end"]]),
                                                                    "label": entity["type"],
                                                                    "start": entity["start"],
                                                                    "end": entity["end"]}
            pos_entities.append((entity["start"], entity["end"]))          

        for i in range(len(doc["tokens"])):
            for j in range(1, 4):
                if i+j < len(doc["tokens"]) and (i, i+j) not in pos_entities:
                    neg_entities.append((i, i+j))

        negative_sampling_doc(doc, matched_data, doc_id, neg_entities, lmbda, last_entity_id=entity_id)    

    return matched_data
            

def negative_sampling_doc(doc, matched_data, doc_id, neg_entities, lmbda, last_entity_id):

    sample_size = math.ceil(lmbda * len(doc["tokens"]))
    neg_sample = random.sample(neg_entities, sample_size)

    entity_id = last_entity_id
    for (i,j) in neg_sample:
        entity_id += 1
        matched_data[f"doc_{doc_id}"][f"entity_{entity_id}"] = {"text": " ".join(doc["tokens"][i:j]), 
                                                                "label": "O",
                                                                "start": i,
                                                                "end": j}                          


def similarity_embedding(matched_data):
    
    bert_config = BertConfig.from_pretrained('bert-large-cased')         
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_embeddings = DiffusionNERBertEmbeddings(bert_config)

    for doc_id in matched_data:
        entities_embds = torch.empty(len(matched_data[doc_id]), 1024)       # findet speicherung/zugriff in folge richtig statt?

        i = 0
        for entity_id in matched_data[doc_id]:
            entity_token_ids = torch.tensor(bert_tokenizer.encode(matched_data[doc_id][entity_id]["text"], add_special_tokens=True)).unsqueeze(0)
            with torch.no_grad():
                entity_embedding = bert_embeddings(input_ids=entity_token_ids)
            entity_cls_embedding = entity_embedding[:, 0, :].unsqueeze(0)                   # nochmal überprüfen

            matched_data[doc_id][entity_id]["embedding"] = entity_cls_embedding
            matched_data[doc_id][entity_id]["index"] = i
            entities_embds[i] = entity_cls_embedding
            i+=1
        
        matched_data[doc_id]["embeddings"] = entities_embds


def knn(matched_data, k):

    for doc_id in matched_data:
        for entity_id in matched_data[doc_id]:
            if entity_id != "embeddings":
                entity_dist_vector = torch.nn.functional.cosine_similarity(matched_data[doc_id][entity_id]["embedding"], matched_data[doc_id]["embeddings"])
                _, entity_knn_indices = entity_dist_vector.topk(k, largest=True)

                entity_knn_labels = []
                entity_knn_indices_list = entity_knn_indices.tolist()[0]
                for entity_id_other in matched_data[doc_id]:
                    if entity_id_other != "embeddings":
                        if matched_data[doc_id][entity_id_other]["index"] in entity_knn_indices_list:
                            entity_knn_labels.append(matched_data[doc_id][entity_id_other]["label"])
                            entity_knn_indices_list.remove(matched_data[doc_id][entity_id_other]["index"])

                if len(entity_knn_labels) != 0:
                    counter = Counter(entity_knn_labels)
                    majority_label, _ = counter.most_common(1)[0]

                    matched_data[doc_id][entity_id]["label"] = majority_label         # konditionieren? - weniger write/runtime?


def write_json(matched_data, data, data_file_out):

    # vergabe von orig_id überprüfen

    doc_id = 0
    for doc in data:
        doc_id+=1
        doc["orig_id"] = str(doc_id)

        doc["entities"] = []
        for entity_id in matched_data[f"doc_{doc_id}"]:
            if entity_id != "embeddings":
                if matched_data[f"doc_{doc_id}"][entity_id]["label"] != "O":
                    doc["entities"].append({"start": matched_data[f"doc_{doc_id}"][entity_id]["start"],
                                            "end": matched_data[f"doc_{doc_id}"][entity_id]["end"],
                                            "type": matched_data[f"doc_{doc_id}"][entity_id]["label"]})
        
    with open(data_file_out, 'w') as data_file_out:
        json.dump(data, data_file_out, indent=4)



def label_retrieval(data_file_in, data_file_out):

    data = read_json(data_file_in)
    matched_data = match_entities(data, lmbda=0)            # Menge von der gesampled wird sehr klein -> Parameter -> oder globaleres Sampling
    similarity_embedding(matched_data)
    knn(matched_data, k=3)
    write_json(matched_data, data, data_file_out)


def main():
    data_file_in = "./noisebench/conll03_clean_train.json"
    data_file_out = "./noisebench/conll03_clean_train_lr.json"
    label_retrieval(data_file_in, data_file_out)


if __name__ == "__main__":
    main()