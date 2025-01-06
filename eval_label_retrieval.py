import label_retrieval as lr
import json
from seqeval.metrics import f1_score, classification_report, accuracy_score


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


def extract_label_seq(matched_entities, no_docs):
    
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


def get_label_sequence(data_file):
    
    data = read_json(data_file)

    matched_entities, no_docs = match_entities(data)

    return extract_label_seq(matched_entities, no_docs), no_docs


def get_retrieved_label_sequences(runs, labelset):

    retrieved_labels_seqs_cls = [None] * runs
    retrieved_labels_seqs_sum = [None] * runs
    for i in range(1, runs+1):
        retrieved_file_cls = f"./results/{labelset}_cls_{i}.json"
        retrieved_labels_seq_cls, _ = get_label_sequence(retrieved_file_cls)
        retrieved_labels_seqs_cls[i-1] = retrieved_labels_seq_cls
        
        retrieved_file_sum = f"./results/{labelset}_sum_{i}.json"
        retrieved_labels_seq_sum, _ = get_label_sequence(retrieved_file_sum)
        retrieved_labels_seqs_sum[i-1] = retrieved_labels_seq_sum
    
    return retrieved_labels_seqs_cls, retrieved_labels_seqs_sum


def write_parameters(labelset, split, k, lmbda):

    with open(f"./results/results_{labelset}_k={k}.txt", "w") as result_file:
        result_file.write("PARAMETERS: \n \n")
        result_file.write(f"labelset: {labelset} \n")
        result_file.write(f"split: {split} \n")
        result_file.write(f"lambda: {lmbda} \n")
        result_file.write(f"k: {k} \n \n \n")


def write_scores(labelset, k, clean_labels, original_labels, retrieved_labels_cls, retrieved_labels_sum):

    with open(f"./results/results_{labelset}_k={k}.txt", "a") as result_file:
        result_file.write("CLS: \n \n")

        for i in range(1,4):
            result_file.write(f"Run {i}: \n \n")
            result_file.write(f"File: ./results/{labelset}_cls_{i}.json \n \n")
            result_file.write(f"F1: {f1_score(clean_labels, retrieved_labels_cls[i-1])} \n \n")
            result_file.write(f"{classification_report(clean_labels, retrieved_labels_cls[i-1])} \n")
            result_file.write(f"Relabeling rate: {1-accuracy_score(original_labels, retrieved_labels_cls[i-1])} \n \n \n")

        result_file.write("SUM: \n \n")

        for i in range(1,4):
            result_file.write(f"Run {i}: \n \n")
            result_file.write(f"File: ./results/{labelset}_sum_{i}.json \n \n")
            result_file.write(f"F1: {f1_score(clean_labels, retrieved_labels_sum[i-1])} \n \n")
            result_file.write(f"{classification_report(clean_labels, retrieved_labels_sum[i-1])} \n")
            result_file.write(f"Relabeling rate: {1-accuracy_score(original_labels, retrieved_labels_sum[i-1])} \n \n \n")


def relabeling_report(original_file, labelset, k, no_docs, original_labels, retrieved_labels_cls, retrieved_labels_sum):

    original_data = read_json(original_file)

    with open(f"./results/results_{labelset}_k={k}.txt", "a") as result_file:
        result_file.write("QUALITATIVE RELABELING EVALUATION: \n \n")

        for i in range(no_docs):
            if original_labels[i] != retrieved_labels_cls[0][i] or original_labels[i] != retrieved_labels_sum[0][i]:
                result_file.write("{:<20} {:<20} {:<20} {:<20} \n \n".format("Token", "Original", "CLS", "SUM"))
                    
                for j in range(len(original_data[i]["tokens"])):
                    result_file.write("{:<20} {:<20} {:<20} {:<20} \n".format(f"{original_data[i]["tokens"][j]}", f"{original_labels[i][j]}", f"{retrieved_labels_cls[0][i][j]}", f"{retrieved_labels_sum[0][i][j]}"))
                    
                result_file.write("\n \n")



def eval(labelset, split, k, lmbda, clean_file, original_file, runs):

    write_parameters(labelset, split, k, lmbda)

    clean_labels, no_docs = get_label_sequence(clean_file)
    original_labels, _ = get_label_sequence(original_file)

    retrieved_labels_cls, retrieved_labels_sum = get_retrieved_label_sequences(runs, labelset)

    write_scores(labelset, k, clean_labels, original_labels, retrieved_labels_cls, retrieved_labels_sum)

    relabeling_report(original_file, labelset, k, no_docs, original_labels, retrieved_labels_cls, retrieved_labels_sum)


def run(data_file_in, labelset, lmbda, k):

    for i in range(1,4):
        data_file_out = f"./results/{labelset}_cls_{i}.json"
        print(f"Run {i} CLS started.")
        lr.label_retrieval(data_file_in, data_file_out, "cls", lmbda, k)

    for i in range(1,4):
        data_file_out = f"./results/{labelset}_sum_{i}.json"
        print(f"Run {i} SUM started.")
        lr.label_retrieval(data_file_in, data_file_out, "sum", lmbda, k)


def main():

    data_file_in = "./noisebench/conll03_noisy_mv_oracle_train.json"
    labelset = "mv_oracle"
    lmbda = 0.33
    k = 3

    run(data_file_in, labelset, lmbda, k)
    
    split = "train"
    runs = 3
    clean_file = "./noisebench/conll03_clean_train.json"

    eval(labelset, split, k, lmbda, clean_file, data_file_in, runs)


if __name__ == "__main__":
    main()