def get_data_csv_file(filename):
    """Read data from CSV file."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append([float(x) for x in line.strip().split(',')])
    return data

def print_best(accu_data, sigmoid_data, softmax_data, output_file):
    """Print best values to file."""
    with open(output_file, 'w') as f:
        f.write("Best accuracy: {}\n".format(max(x[1] for x in accu_data)))
        f.write("Best sigmoid: {}\n".format(max(x[1] for x in sigmoid_data)))
        f.write("Best softmax: {}\n".format(max(x[1] for x in softmax_data)))