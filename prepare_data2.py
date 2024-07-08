import argparse
import pickle
import re
import os
from scipy.io import wavfile
import numpy as np

def extract_params(filename):
    match = re.search(r"_r(\d+)_c(\d+)", filename)
    if match:
        resonance = int(match.group(1))
        cutoff = int(match.group(2))
        return resonance, cutoff
    else:
        raise ValueError(f"Filename {filename} does not contain resonance and cutoff information in the expected format.")

def normalize_params(params, max_value=127):
    return params / max_value

def process_file(in_file, out_file, sample_time, sample_rate):
    in_rate, in_data = wavfile.read(in_file)
    out_rate, out_data = wavfile.read(out_file)
    assert in_rate == out_rate, "in_file and out_file must have same sample rate"

    sample_size = int(sample_rate * sample_time)
    length = len(in_data) - len(in_data) % sample_size

    x = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)
    y = out_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    resonance, cutoff = extract_params(os.path.basename(in_file))
    resonance_norm, cutoff_norm = normalize_params(np.array([resonance, cutoff]))

    params = np.tile(np.array([cutoff_norm, resonance_norm]), (len(x), 1)).astype(np.float32)

    return x, y, params

def main(args):
    dry_files = sorted([os.path.join(args.dry_path, f) for f in os.listdir(args.dry_path) if f.endswith('.wav')])
    wet_files = sorted([os.path.join(args.wet_path, f) for f in os.listdir(args.wet_path) if f.endswith('.wav')])

    assert len(dry_files) == len(wet_files), "Number of dry files must match number of wet files"

    x_list, y_list, params_list = [], [], []

    for in_file, out_file in zip(dry_files, wet_files):
        x, y, params = process_file(in_file, out_file, args.sample_time, args.sample_rate)
        x_list.append(x)
        y_list.append(y)
        params_list.append(params)

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    params = np.concatenate(params_list, axis=0)

    split = lambda d: np.split(d, [int(len(d) * 0.6), int(len(d) * 0.8)])

    d = {}
    d["x_train"], d["x_valid"], d["x_test"] = split(x)
    d["y_train"], d["y_valid"], d["y_test"] = split(y)
    d["params_train"], d["params_valid"], d["params_test"] = split(params)
    d["mean"], d["std"] = d["x_train"].mean(), d["x_train"].std()

    for key in "x_train", "x_valid", "x_test":
        d[key] = (d[key] - d["mean"]) / d["std"]

    pickle.dump(d, open(args.data, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dry_path", help="Path to the directory containing dry (input) signals")
    parser.add_argument("wet_path", help="Path to the directory containing wet (output) signals")
    parser.add_argument("--data", default="data_test_params.pickle")
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("--sample_rate", type=int, default=44100, help="Sample rate of the audio files")
    args = parser.parse_args()
    main(args)
