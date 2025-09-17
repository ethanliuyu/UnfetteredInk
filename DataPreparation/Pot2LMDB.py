import json
import os
import pickle
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed

import lmdb
import numpy as np

noise = 5

charList = None
charListPath = "./charlist.json"
with open(charListPath, 'r') as f:
    charList = json.load(f)
print(charList)
env_path = './LMDB/lmdb'
pot_dir = "./PotData/"
outPath = "./LMDB/"


def glyph_to_svg(svgPath):
    # This function remains unchanged.
    pass  # Placeholder for brevity


def one_file(f):
    while True:
        header = np.fromfile(f, dtype='uint8', count=8)
        if not header.size:
            break

        sample_size = header[0] + (header[1] << 8)

        raw = header[2:6].tobytes()
        tagcode = int.from_bytes(raw, byteorder='little', signed=False)

        stroke_num = header[6] + (header[7] << 8)

        traj = []

        for i in range(stroke_num):
            while True:
                header = np.fromfile(f, dtype='int16', count=2)
                x, y = header[0], header[1]
                if x == -1 and y == 0:
                    traj.append([100000, 100000])
                    break
                else:
                    traj.append([x, y])

        header = np.fromfile(f, dtype='int16', count=2)

        pts = np.array(traj)

        pts = np.round(pts * 0.9).astype(int)
        xmin = np.min(pts[:, 0])
        pts[:, 0] -= xmin

        sorted_col1 = np.unique(np.sort(pts[:, 0]))
        xmax = sorted_col1[-2]
        xmin = sorted_col1[0]

        ymin = np.min(pts[:, 1])
        pts[:, 1] -= ymin

        sorted_col1 = np.unique(np.sort(pts[:, 1]))
        ymax = sorted_col1[-2]
        ymin = sorted_col1[0]

        maxImage = 256

        translate_x = abs((maxImage - (xmax - xmin)) // 2)
        translate_y = abs((maxImage - (ymax - ymin)) // 2)

        pts[:, 0] += translate_x
        pts[:, 1] += translate_y

        command = []
        para = []

        paths = ""

        path = "M {} {} \n".format(pts[0][0], pts[0][1])
        command.append("M")
        para.append([pts[0][0], pts[0][1]])

        paths += path
        stroke_start_tag = False

        for i in range(1, len(pts)):
            if pts[i][0] > 1000 and pts[i][1] > 1000:
                stroke_start_tag = True
                continue

            if pts[i][0] > maxImage - noise - 1 or pts[i][1] > maxImage - noise - 1:
                pts[i][0] = maxImage - noise - 1
                pts[i][1] = maxImage - noise - 1
            if pts[i][0] < noise + 1 or pts[i][1] < noise + 1:
                pts[i][0] = noise + 1
                pts[i][1] = noise + 1

            if stroke_start_tag:
                path = "M {} {} \n".format(pts[i][0], pts[i][1])
                command.append("M")
                para.append([pts[i][0], pts[i][1]])
                stroke_start_tag = False
            else:
                if pts[i][0] < 0 or pts[i][1] < 0:
                    continue  # Skip invalid points
                path = "L {} {} \n".format(pts[i][0], pts[i][1])
                command.append("L")
                para.append([pts[i][0], pts[i][1]])

            paths += path

        tempDir = {
            "command": command,
            "para": para
        }
        yield tempDir, tagcode


def process_file(file_name):
    print(f"Processing file: {file_name}")
    env = lmdb.open(env_path, map_size=1024 ** 4, max_dbs=0)
    className = []

    if file_name.endswith('.pot'):
        file_path = os.path.join(pot_dir, file_name)
        className.append(file_name)
        with env.begin(write=True) as txn:
            with open(file_path, 'rb') as f:
                sample_count = 0
                for paths, tagcode in one_file(f):
                    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312', errors='replace')
                    if tagcode_unicode in charList:
                        charName = hex(ord(tagcode_unicode))[2:].upper()
                        key = '{}_{}'.format(file_name, charName)
                        value = pickle.dumps(paths)
                        # Print progress every 1,000 samples
                        sample_count += 1
                        if sample_count % 1000 == 0:
                            print(f"{file_name}: Processed {sample_count} samples")

                        txn.put(key.encode('utf-8'), value)
    env.close()
    print(f"Finished processing file: {file_name}")
    return className


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed

    files_to_process = [file_name for file_name in os.listdir(pot_dir) if file_name.endswith('.pot')]
    className = []

    # Modify the list to only include the first K files
    files_to_process = files_to_process[:10]

    total_files = len(files_to_process)
    print(f"Starting processing of {total_files} .pot files...")

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_name): file_name for file_name in files_to_process}
        completed_files = 0
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                result = future.result()
                className.extend(result)
                completed_files += 1
                print(f"Completed processing file: {file_name} ({completed_files}/{total_files})")
            except Exception as exc:
                print(f'{file_name} generated an exception: {exc}')

    # Write the className to LMDB
    env = lmdb.open(env_path, map_size=1024 ** 4)
    with env.begin(write=True) as txn:
        key = "className"
        value = pickle.dumps(className)
        txn.put(key.encode('utf-8'), value)
    env.close()

    print("Processing complete.")

