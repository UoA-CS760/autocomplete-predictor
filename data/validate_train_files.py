import os
import shutil
import time

# checks that training files exist
# note: some files may be the same name such as __init__.py
# be sure to give the files a unique name if copying into one directory

training_files_list = 'python10k_train.txt'

with open(training_files_list, 'r') as f:
    print(f'loading file list {training_files_list}')

    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    lines = [line.replace('\\', '/') for line in lines]

found_count = 0
for line in lines:
    fp = os.path.join(os.getcwd(), line)

    if not os.path.exists(fp):
        print(f"error! file does not exist {fp}")
    else:
        found_count += 1

print(f'found {found_count}/{len(lines)} files')

train_file_dir = os.path.join(os.getcwd(), 'train_files')

copy_files_cmd = input(f'Copy train files to {train_file_dir}?\nWarning! Will remove existing files in {train_file_dir} [y]: ').strip().lower()
copy_files = (copy_files_cmd == '') or (copy_files_cmd == 'y')

if copy_files:
    # remove existing folder
    if os.path.exists(train_file_dir):
        shutil.rmtree(train_file_dir)

    os.makedirs(train_file_dir)

    copy_count = 0
    for line in lines:
        src_fp = os.path.join(os.getcwd(), line)
        filename = line.replace('/', '_') # unique filename
        dest_fp = os.path.join(train_file_dir, filename)

        shutil.copyfile(src_fp, dest_fp)
        copy_count += 1

        if not os.path.exists(dest_fp):
            print('not found!', dest_fp)

    print(f'{copy_count} files copied')
