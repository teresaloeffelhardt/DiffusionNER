import json
import math
import random
from transformers import BertConfig, BertTokenizer
from diffusionner.modeling_bert import BertEmbeddings as DiffusionNERBertEmbeddings
import torch
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.datasets.sequence_labeling import NER_NOISEBENCH
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
from collections import Counter

# read json content and structure into data
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
                    if not any(not(l<=i or k>=i+j) for (k,l) in pos_entities):
                        neg_entities.append((i, i+j))

        if lmbda > 0:
            matched_data_nes = negative_sampling_doc(doc, matched_data, doc_id, neg_entities, lmbda, last_entity_id=entity_id)
        else:
            matched_data_nes = matched_data

        if doc["orig_id"] in origins:
            origins[doc["orig_id"]]["docs"][f"doc_{doc_id}"] = doc["tokens"]
            origins[doc["orig_id"]]["n_entities"] += len(matched_data_nes[f"doc_{doc_id}"])-1
        else:
            origins[doc["orig_id"]] = {"docs": {f"doc_{doc_id}": doc["tokens"]},
                                       "n_entities": len(matched_data_nes[f"doc_{doc_id}"])-1}

    return matched_data_nes, origins
            

def negative_sampling_doc(doc, matched_data, doc_id, neg_entities, lmbda, last_entity_id):

    # sample_size = math.ceil(lmbda * len(doc["tokens"]))                     
    
    n_entities = len(matched_data[f"doc_{doc_id}"])-1                          # kann in ungelabelten Sätzen auch nicht mehr labeln
    sample_size_float = lmbda * n_entities
    sample_size_int = int(sample_size_float)
    sample_size_frac = sample_size_float-sample_size_int
    if random.random() < sample_size_frac:                                    
        sample_size = sample_size_int+1
    else:
        sample_size = sample_size_int

    neg_entities_size = len(neg_entities)
    if sample_size > neg_entities_size:
        sample_size = neg_entities_size                                  
    neg_sample = random.sample(neg_entities, sample_size)

    left_neg_entities = list(set(neg_entities)-set(neg_sample))          # in diesem Abschnitt direkt mit sets arbeiten und erst am Ende wieder Conversion?
    removed = -1
    while left_neg_entities and removed!=0:
        removed = 0
        neg_sample_new = []
        for (i,j) in neg_sample:
            if not any(not(k==i and l==j) and not(l<=i or k>=j) for (k,l) in neg_sample_new):
                neg_sample_new.append((i,j))
                removed+=1
        neg_sample = neg_sample_new
        if removed > len(left_neg_entities):
            removed = len(left_neg_entities)
        add_neg_sample = random.sample(left_neg_entities, removed)
        left_neg_entities = list(set(left_neg_entities)-set(add_neg_sample))
        neg_sample += add_neg_sample
    
    neg_sample_new = []
    for (i,j) in neg_sample:
        if not any(not(k==i and l==j) and not(l<=i or k>=j ) for (k,l) in neg_sample_new):
            neg_sample_new.append((i,j))
    neg_sample = neg_sample_new

    entity_id = last_entity_id
    for (i,j) in neg_sample:
        entity_id += 1
        matched_data[f"doc_{doc_id}"][f"entity_{entity_id}"] = {"text": " ".join(doc["tokens"][i:j]), 
                                                                "label": "O",
                                                                "start": i,
                                                                "end": j}    

    return matched_data


def sort_entities(matched_data, origins):

    for doc_id in matched_data:
        doc = {"orig_id": matched_data[doc_id]["orig_id"]}

        n_entities_doc = len(matched_data[doc_id])-1
        tokens = len(origins[matched_data[doc_id]["orig_id"]]["docs"][doc_id])
        last_end = 0
        min_start = tokens
        min_entity_id = "entity_0"
        for i in range(n_entities_doc):
            for entity_id in matched_data[doc_id]:
                if entity_id != "orig_id":
                    start = matched_data[doc_id][entity_id]["start"]

                    if start>=last_end and start<min_start:
                        min_start = start
                        min_entity_id = entity_id

            doc[f"entity_{i+1}"] = matched_data[doc_id][min_entity_id]
            last_end = matched_data[doc_id][min_entity_id]["end"]
            min_start = tokens
            min_entity_id = "entity_0"
        
        matched_data[doc_id] = doc

    return matched_data


def similarity_embedding_cls_difner(matched_data, origins):
    
    bert_config = BertConfig.from_pretrained('bert-large-cased')         
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    bert_embeddings = DiffusionNERBertEmbeddings(bert_config)                           

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
    
    return matched_data, origins


def similarity_embedding_cls(labelset, fine_tune, matched_data, origins, ft_epochs):
    
    if fine_tune:
        bert_embeddings = TransformerWordEmbeddings('bert-large-cased', is_document_embedding=True, subtoken_pooling='mean', fine_tune=True) # use_context for fine-tuning?
    else:
        bert_embeddings = TransformerDocumentEmbeddings('bert-large-cased', subtoken_pooling='mean', fine_tune=True) # use_context for fine-tuning?

    # TransformerWordEmbeddings + .document_cls_pooling ?
    # TransformerWordEmbeddings + is_document_embedding ?
    # kombination ? - nochmal checken, ob das valide ist & das CLS-embedding ausgibt
    
    if fine_tune:
        print("\t Fine-Tuning started.")
        corpus = NER_NOISEBENCH(noise=labelset)                  
        label_dict = corpus.make_label_dictionary(label_type='ner', add_unk=False)
        tagger = SequenceTagger(hidden_size=256,
                            embeddings=bert_embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=False,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )
        trainer = ModelTrainer(tagger, corpus)
        trainer.fine_tune(f'results/fine_tune_{labelset}',     
                    learning_rate=5.0e-6,
                    mini_batch_size=4,
                    mini_batch_chunk_size=1,
                    max_epochs=ft_epochs,                       
                    )
        print("\t Fine-Tuning completed.")

    for orig_id in origins:
        origins[orig_id]["embeddings"] = torch.zeros(origins[orig_id]["n_entities"], 1, 1024)

        i = 0
        for doc_id in origins[orig_id]["docs"]:
            for entity_id in matched_data[doc_id]:
                if entity_id != "orig_id":
                    entity_text = matched_data[doc_id][entity_id]["text"]
                    entity = Sentence(entity_text)

                    bert_embeddings.embed(entity)
                    entity_embedding = entity.embedding

                    matched_data[doc_id][entity_id]["embedding"] = entity_embedding
                    matched_data[doc_id][entity_id]["index"] = i
                    origins[orig_id]["embeddings"][i] = entity_embedding
                    i+=1
    
    return matched_data, origins


def similarity_embedding_sum(labelset, fine_tune, matched_data, origins, ft_epochs):       

    bert_embeddings = TransformerWordEmbeddings('bert-large-cased', subtoken_pooling='mean', fine_tune=True)

    if fine_tune:
        print("\t Fine-Tuning started.")
        corpus = NER_NOISEBENCH(noise=labelset)                  # make generally usable - auf clean fine-tunen?
        label_dict = corpus.make_label_dictionary(label_type='ner', add_unk=False)
        tagger = SequenceTagger(hidden_size=256,
                            embeddings=bert_embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=False,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )
        trainer = ModelTrainer(tagger, corpus)
        trainer.fine_tune(f'results/fine_tune_{labelset}',      
                    learning_rate=5.0e-6,
                    mini_batch_size=4,
                    mini_batch_chunk_size=1,
                    max_epochs=ft_epochs,                        
                    )                                    
        print("\t Fine-Tuning completed.")

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

    return matched_data, origins


def knn(matched_data, origins, k):

    for orig_id in origins:
        for doc_id in origins[orig_id]["docs"]:
            for entity_id in matched_data[doc_id]:
                if entity_id != "orig_id":
                    entity_dist_vector = torch.zeros(len(origins[orig_id]["embeddings"]))
                    for i in range(len(origins[orig_id]["embeddings"])):
                        entity_dist_vector[i] = torch.nn.functional.cosine_similarity(matched_data[doc_id][entity_id]["embedding"], origins[orig_id]["embeddings"][i])

                    if entity_dist_vector.shape[0] >= k:
                        _, entity_knn_indices = entity_dist_vector.topk(k, largest=True)

                        # neighborhood printing
                        neighborhood = [matched_data[doc_id][entity_id]["text"]]

                        entity_knn_indices_list = entity_knn_indices.tolist()
                        entity_knn_labels = []
                        for doc_id_other in origins[orig_id]["docs"]:
                            for entity_id_other in matched_data[doc_id_other]:
                                if entity_id_other != "orig_id":
                                    if matched_data[doc_id_other][entity_id_other]["index"] in entity_knn_indices_list:

                                        # neighborhood printing
                                        neighborhood.append(matched_data[doc_id_other][entity_id_other]["text"])

                                        entity_knn_labels.append(matched_data[doc_id_other][entity_id_other]["label"])
                                        entity_knn_indices_list.remove(matched_data[doc_id_other][entity_id_other]["index"])

                        if len(entity_knn_labels) != 0:
                            counter = Counter(entity_knn_labels)
                            majority_label, _ = counter.most_common(1)[0]

                            matched_data[doc_id][entity_id]["label"] = majority_label         # konditionieren? - weniger write/runtime?
                        
                        # neighborhood printing
                        print(neighborhood)

    return matched_data


def write_json(matched_data, data, data_file_out):

    doc_id = 0
    for doc in data:
        doc_id+=1

        doc_entities = []
        for entity_id in matched_data[f"doc_{doc_id}"]:
            if entity_id != "embeddings" and entity_id != "orig_id":
                if matched_data[f"doc_{doc_id}"][entity_id]["label"] != "O":
                    doc_entities.append({"start": matched_data[f"doc_{doc_id}"][entity_id]["start"],
                                            "end": matched_data[f"doc_{doc_id}"][entity_id]["end"],
                                            "type": matched_data[f"doc_{doc_id}"][entity_id]["label"]})
                    
        doc["entities"] = doc_entities
        
    with open(data_file_out, 'w') as data_file_out:
        json.dump(data, data_file_out, indent=4)    



def label_retrieval(data_file_in, data_file_out, similarity_embedding, lmbda, k, labelset, fine_tune, ft_epochs):

    data = read_json(data_file_in)
    matched_data, origins = match_entities(data, lmbda)
    matched_data_sorted = sort_entities(matched_data, origins)
    if similarity_embedding == "cls":
        matched_data_embd, origins_embd = similarity_embedding_cls(labelset, fine_tune, matched_data_sorted, origins, ft_epochs)
    elif similarity_embedding == "sum":
        matched_data_embd, origins_embd = similarity_embedding_sum(labelset, fine_tune, matched_data_sorted, origins, ft_epochs)
    matched_data_knn = knn(matched_data_embd, origins_embd, k)
    write_json(matched_data_knn, data, data_file_out)


def main():
    # veraltet
    data_file_in = "./noisebench/conll03_noisy_original_train.json"
    data_file_out = "./results/original_cls_debugging.json"
    label_retrieval(data_file_in, data_file_out, "cls", 0.33, 3)


if __name__ == "__main__":
    main()