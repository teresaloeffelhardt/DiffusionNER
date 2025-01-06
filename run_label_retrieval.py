import label_retrieval as lr


data_file_in = "./noisebench/conll03_noisy_original_train.json"

lmbda = 0.33
k = 3
   
for i in range(1,4):
    data_file_out = f"./results/original_sum_{i}.json"

    print(f"Run {i} started.")
    lr.label_retrieval(data_file_in, data_file_out, "sum", lmbda, k)