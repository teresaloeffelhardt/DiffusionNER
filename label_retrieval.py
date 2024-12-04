import json
import math
import random
from transformers import BertConfig, BertTokenizer
from diffusionner.modeling_bert import BertEmbeddings as DiffusionNERBertEmbeddings
import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from collections import Counter

import pprint


def read_json(data_file):
    
    with open(data_file, 'r') as data_file:
        data = json.load(data_file)
    
    return data


def match_entities(data, lmbda):

    matched_data = {}
    origins = {}

    doc_id = 0
    for doc in data:
        doc_id += 1
        matched_data[f"doc_{doc_id}"] = {"orig_id": doc["orig_id"]}
        
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

        if doc["orig_id"] in origins:
            origins[doc["orig_id"]]["docs"][f"doc_{doc_id}"] = doc["tokens"]
            origins[doc["orig_id"]]["n_entities"] += len(matched_data[f"doc_{doc_id}"])-1
        else:
            origins[doc["orig_id"]] = {"docs": {f"doc_{doc_id}": doc["tokens"]},
                                       "n_entities": len(matched_data[f"doc_{doc_id}"])-1}

    return matched_data, origins
            

def negative_sampling_doc(doc, matched_data, doc_id, neg_entities, lmbda, last_entity_id):

    sample_size = math.ceil(lmbda * len(doc["tokens"]))
    neg_entities_size = len(neg_entities)
    if sample_size > neg_entities_size:
        sample_size = neg_entities_size                                  #  besseren Weg bzw. lmbda*neg_entities_size immer nehmen?
    neg_sample = random.sample(neg_entities, sample_size)

    entity_id = last_entity_id
    for (i,j) in neg_sample:
        entity_id += 1
        matched_data[f"doc_{doc_id}"][f"entity_{entity_id}"] = {"text": " ".join(doc["tokens"][i:j]), 
                                                                "label": "O",
                                                                "start": i,
                                                                "end": j}                          


def similarity_embedding_cls(matched_data, origins):
    
    bert_config = BertConfig.from_pretrained('bert-large-cased')         
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_embeddings = DiffusionNERBertEmbeddings(bert_config)                           # auch die von flair? -> andere Ergebnisse? 
                                                                                        # benutze ich die richtig? mit entity level etc.

    for orig_id in origins:
        origins[orig_id]["embeddings"] = torch.zeros(origins[orig_id]["n_entities"], 1, 1024)

        i = 0
        for doc_id in origins[orig_id]["docs"]:
            for entity_id in matched_data[doc_id]:
                if entity_id != "orig_id":
                    entity_token_ids = torch.tensor(bert_tokenizer.encode(matched_data[doc_id][entity_id]["text"], add_special_tokens=True)).unsqueeze(0)
                    with torch.no_grad():
                        entity_embedding = bert_embeddings(input_ids=entity_token_ids)
                    entity_cls_embedding = entity_embedding[:, 0, :]

                    matched_data[doc_id][entity_id]["embedding"] = entity_cls_embedding
                    matched_data[doc_id][entity_id]["index"] = i
                    origins[orig_id]["embeddings"][i] = entity_cls_embedding
                    i+=1


def similarity_embedding_sum(matched_data, origins):       

    bert_embeddings = TransformerWordEmbeddings('bert-large-cased', subtoken_pooling='mean')      # brauche embedding mit kontext

    for orig_id in origins:
        origins[orig_id]["embeddings"] = torch.zeros(origins[orig_id]["n_entities"], 1, 1024)       

        i = 0                                                
        for doc_id in origins[orig_id]["docs"]:
            doc_tokens = origins[orig_id]["docs"][doc_id]
            doc_text = " ".join(doc_tokens)
            doc = Sentence(doc_text)
            
            bert_embeddings.embed(doc)

            for entity_id in matched_data[doc_id]:
                if entity_id != "orig_id":
                    entity_sum_embedding = torch.zeros(1, 1024)
                    for k in range(matched_data[doc_id][entity_id]["start"], matched_data[doc_id][entity_id]["end"]):
                        entity_sum_embedding[0] += doc[k].embedding   

                    matched_data[doc_id][entity_id]["embedding"] = entity_sum_embedding
                    matched_data[doc_id][entity_id]["index"] = i
                    origins[orig_id]["embeddings"][i] = entity_sum_embedding
                    i+=1


def knn(matched_data, origins, k):

    for orig_id in origins:
        for doc_id in origins[orig_id]["docs"]:
            for entity_id in matched_data[doc_id]:
                if entity_id != "orig_id":
                    entity_dist_vector = torch.nn.functional.cosine_similarity(matched_data[doc_id][entity_id]["embedding"], origins[orig_id]["embeddings"])
                    if entity_dist_vector.shape[0] >= k:
                        _, entity_knn_indices = entity_dist_vector.topk(k, largest=True)

                        entity_knn_labels = []
                        entity_knn_indices_list = entity_knn_indices.tolist()   # [0] - etwas ist komisch
                        for doc_id_other in origins[orig_id]["docs"]:
                            for entity_id_other in matched_data[doc_id_other]:
                                if entity_id_other != "orig_id":
                                    if matched_data[doc_id_other][entity_id_other]["index"] in entity_knn_indices_list:
                                        entity_knn_labels.append(matched_data[doc_id_other][entity_id_other]["label"])
                                        entity_knn_indices_list.remove(matched_data[doc_id_other][entity_id_other]["index"])

                        if len(entity_knn_labels) != 0:
                            counter = Counter(entity_knn_labels)
                            majority_label, _ = counter.most_common(1)[0]

                            matched_data[doc_id][entity_id]["label"] = majority_label         # konditionieren? - weniger write/runtime?


def write_json(matched_data, data, data_file_out):

    doc_id = 0
    for doc in data:
        doc_id+=1

        doc["entities"] = []
        for entity_id in matched_data[f"doc_{doc_id}"]:
            if entity_id != "embeddings" and entity_id != "orig_id":
                if matched_data[f"doc_{doc_id}"][entity_id]["label"] != "O":
                    doc["entities"].append({"start": matched_data[f"doc_{doc_id}"][entity_id]["start"],
                                            "end": matched_data[f"doc_{doc_id}"][entity_id]["end"],
                                            "type": matched_data[f"doc_{doc_id}"][entity_id]["label"]})
        
    with open(data_file_out, 'w') as data_file_out:
        json.dump(data, data_file_out, indent=4)



def label_retrieval(data_file_in, data_file_out):

    data = read_json(data_file_in)
    matched_data, origins = match_entities(data, lmbda=0.35)            # Menge von der gesampled wird sehr klein -> Parameter -> oder globaleres Sampling
    similarity_embedding_sum(matched_data, origins)
    knn(matched_data, origins, k=3)
    write_json(matched_data, data, data_file_out)


def main():
    data_file_in = "./test.json"
    data_file_out = "./results/test_lr_sum.json"
    label_retrieval(data_file_in, data_file_out)


if __name__ == "__main__":
    main()