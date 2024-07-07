import pickle

def read_pickle_file(file_path, lines_to_print=5):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Check if the data is a list or another type
    if isinstance(data, list):
        # If it's a list, print the first few items
        for i, item in enumerate(data[:lines_to_print]):
            print(f"Line {i + 1}: {item}")
    elif isinstance(data, dict):
        # If it's a dictionary, print the first few key-value pairs
        for i, (key, value) in enumerate(data.items()):
            if i >= lines_to_print:
                break
            print(f"Key {i + 1}: {key}, Value: {value}")
    else:
        print(f"Unsupported data type: {type(data)}")
        print(f"Data: {data}")

# Use forward slashes to avoid issues with backslashes in file path
read_pickle_file('../data.pickle', lines_to_print=2)



