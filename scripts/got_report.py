
import sys
import shutil


def report(save_file, dirname):
    shutil.make_archive(save_file, 'zip', dirname)


if __name__ == '__main__':
    report(sys.argv[1], sys.argv[2])
