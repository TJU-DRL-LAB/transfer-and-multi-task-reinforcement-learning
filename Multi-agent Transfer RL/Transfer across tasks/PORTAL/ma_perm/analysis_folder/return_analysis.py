import os
import numpy as np
from openpyxl import Workbook

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, help='policy')
    parser.add_argument('--dir', type=str, help='dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    num_of_data = 32
    policy = args.policy
    pre_base_dir = args.dir

    base_dir = os.path.join(pre_base_dir)
    dir = f"{base_dir}/{policy}/return"

    data_files = os.listdir(dir)
    data_files = sorted(data_files, key=lambda x:os.path.getctime(os.path.join(dir, x)))

    rows = []
    for data_file in data_files:
        if not data_file.endswith('npy'):
            continue
        one_row = []
        name = data_file[13:-7]
        one_row.append(name)
        data = np.load(os.path.join(dir, data_file), allow_pickle=True)
        won_list = data.item()['won_list']
        return_list = data.item()['return_list']
        one_row.append(np.mean(won_list))
        one_row.append(np.std(won_list))
        one_row.append(np.mean(return_list))
        one_row.append(np.max(return_list))
        one_row.append(np.std(return_list))
        rows.append(one_row)
    sorted_rows = sorted(rows, key = lambda x:x[3])
    sorted_rows.insert(0, ['env', 'won_rate_mean', 'won_rate_std', 'return_mean', 'return_max', 'return_std'])
    wb = Workbook()
    sheet = wb.active
    for row in sorted_rows:
        sheet.append(row)
    wb.save(f'{dir}/return.xlsx')
    print('finished')