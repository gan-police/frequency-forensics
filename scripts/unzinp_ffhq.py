import os
import glob


def main():
    os.system('pwd')
    ffhq_folder = "./data/ffhq_large/ffhq_1024/"
    ffhq_zip_files = ffhq_folder + "*.zip"

    files = glob.glob(ffhq_zip_files)
    files.sort()

    for file in files:
        folder = file.split('/')[-1].split('-')[0]
        os.system("unzip " + file)
        os.system("mv " + folder + "/*.png "
                  + ffhq_folder)
        os.system("rmdir " + folder)
        os.system("rm " + file)
        print("file done")


if __name__ == '__main__':
    main()
