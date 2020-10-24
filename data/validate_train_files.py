import os
import shutil

# checks that training files exist

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

copy_files_cmd = input(f'copy train files to {train_file_dir}? [y]: ').strip().lower()
copy_files = (copy_files_cmd == '') or (copy_files_cmd == 'y')

if copy_files:
    if not os.path.exists(train_file_dir):
        os.makedirs(train_file_dir)

    copy_count = 0
    for line in lines:
        src_fp = os.path.join(os.getcwd(), line)
        filename = os.path.basename(line)
        dest_fp = os.path.join(train_file_dir, filename)

        if os.path.exists(dest_fp):
            os.remove(dest_fp)

        shutil.copy(src_fp, dest_fp)

        copy_count += 1

    print(f'{copy_count} files copied')
