import os
import argparse
import subprocess


def preprocess_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    block_comment = False
    download_models = False
    new_lines = []
    for i, l in enumerate(lines):
        if l.startswith('#'):
            continue
        if 'get_ipython()' in l:
            continue
        if l.startswith('import urllib.request'):
            download_models = True
            continue
        elif download_models:
            if 'print(\"Something went wrong' in l:
                download_models = False
            continue
        # if block_comment:
        #     if "\"\"\"" in l:

        #         block_comment = False
        l = l.rstrip()
        if len(l) == 0 and (len(new_lines) == 0 or len(new_lines[-1]) == 0):
            continue
        new_lines.append(l)

    new_file = '\n'.join(new_lines)
    with open(filename, 'w') as f:
        f.write(new_file)

    subprocess.run(['autopep8', '-i', filename])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.file), f'File not found: \"{args.file}\"'
    preprocess_file(args.file)