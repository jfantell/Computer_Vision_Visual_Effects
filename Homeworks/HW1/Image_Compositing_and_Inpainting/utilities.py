import os
import shutil
import glob


# Source: https://stackoverflow.com/questions/6489663/find-file-in-folder-without-knowing-the-extension
def get_file_path(input_dir, filename):
    for file in glob.glob(os.path.join(input_dir, f'{filename}.*')):
        if (file.endswith(('png', 'jpg', 'jpeg'))):
            return 1, file
        elif (file.endswith(('.mov', '.avi', '.mp4'))):
            return 2, file
    # No valid files found that match criteria
    return -1, None


def make_dir(dir_):
    if os.path.exists(dir_):
        shutil.rmtree(dir_, ignore_errors=True)
    try:
        os.makedirs(dir_)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise
        pass
