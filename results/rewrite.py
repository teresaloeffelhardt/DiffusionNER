import json

def read_json(data_file):
    
    with open(data_file, 'r') as data_file:
        data = json.load(data_file)
    
    return data


def write_json(data, data_file_out):
        
    with open(data_file_out, 'w') as data_file_out:
        json.dump(data, data_file_out, indent=4)


def main():
    data_file_in = "./noisebench/conll03_noisy_wrench_train.json"
    data_file_out = data_file_in
    data = read_json(data_file_in)
    write_json(data, data_file_out)


if __name__ == "__main__":
    main()