import os
import shutil
import time

# checks that training files exist
# note: some files may be the same name such as __init__.py
# be sure to give the files a unique name if copying into one directory

training_files_list = 'python10k_train.txt'
test_files_list = 'python50_test.txt'


def check_files_exist(files_list):

    found_count = 0
    for line in files_list:
        fp = os.path.join(os.getcwd(), line)

        if not os.path.exists(fp):
            print(f"error! file does not exist {fp}")
        else:
            found_count += 1

    print(f'found {found_count}/{len(files_list)} files')


def load_files(files_list_name):
    with open(files_list_name, 'r') as f:
        print(f'loading file list {files_list_name}')

        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        lines = [line.replace('\\', '/') for line in lines]

    return lines


train_fl = load_files(training_files_list)
test_fl = load_files(test_files_list)

check_files_exist(train_fl)
check_files_exist(test_fl)

train_file_dir = os.path.join(os.getcwd(), 'train_files')
test_file_dir = os.path.join(os.getcwd(), 'test_files')

copy_files_cmd = input(f'Copy train & test files to new directory?\nWarning! Will remove existing files [y]: ').strip().lower()
copy_files = (copy_files_cmd == '') or (copy_files_cmd == 'y')

if copy_files:

    def copy_files(files_list, dest_dir):

        # remove existing folder
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        os.makedirs(dest_dir)

        copy_count = 0
        for line in files_list:
            src_fp = os.path.join(os.getcwd(), line)
            filename = line.replace('/', '_') # unique filename
            dest_fp = os.path.join(dest_dir, filename)

            shutil.copyfile(src_fp, dest_fp)
            copy_count += 1

            if not os.path.exists(dest_fp):
                print('not found!', dest_fp)

        print(f'{copy_count} files copied')

    copy_files(train_fl, train_file_dir)
    copy_files(test_fl, test_file_dir)
