import tarfile
import os
import sys
import glob

# import threading
# from concurrent.futures import ProcessPoolExecutor
# from time import sleep


def untar(name, path, num):
    print('Extracting the %s.tar is starting' %(str(num).zfill(4)))

    tf = tarfile.open(name)
    tf.extractall(path)

    print('Extracting the %s.tar is finished' %(str(num).zfill(4)))


# File path
Data_Dir = '/root/data/Waymo/original/testing'     # training, validation, testing
# File list of *.tar
filename_list = sorted(glob.glob(Data_Dir + "/*.tar"))

print('# of files :', len(filename_list))

threads = []
for filenum, filename in enumerate(filename_list):
    # Save path
    save_path = Data_Dir + '/' + str(filenum).zfill(4)

    # make dir
    os.makedirs(save_path, exist_ok=True)

    # untar
    untar(filename,save_path,filenum)

#     th = threading.Thread( target=untar, args = (filename,save_path,filenum) )
#     # th.daemon = True
#     threads.append(th)
#     th.start()


# # Wait for each thread to complete
# for thread in threads:
#     thread.join()