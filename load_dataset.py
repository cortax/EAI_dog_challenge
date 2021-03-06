import numpy as np
import wget
import os
import tarfile
from PIL import Image
import PIL as pil_image
import glob
import numpy as np
from math import floor

def prepare_dataset():
    if os.path.isdir("./Images"):
        print('Image folder found')
    else:
        if os.path.isfile("images.tar"):
            print('Image archive found')
        else:
            url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
            filename = wget.download(url)
        filename = 'images.tar'
        opener, mode = tarfile.open, 'r'
        file = opener('./' + filename, mode)
        file.extractall()
        file.close()

    cwd = os.getcwd()
    os.makedirs(cwd + '/Images/train', exist_ok=True)
    os.makedirs(cwd + '/Images/test', exist_ok=True)

    for d in [x[0] for x in os.walk('./Images/')]:
        basedir = os.path.split(d)
        if not basedir[1] == '' and not basedir[1] == 'train' and not basedir[1] == 'test' and basedir[0] == './Images':
            print(basedir[1])
            L = glob.glob(str(d) + '/*.jpg')
            lastfile = floor(0.85*len(L))
            for pathname in L[0:lastfile]:
                parsed_filename = pathname.split('/')
                parsed_filename[0] = cwd
                oldpathname = '/'.join(parsed_filename)
                parsed_filename.insert(2, 'train')
                newpathname = '/'.join(parsed_filename)
                os.makedirs('/'.join(parsed_filename[:-1]), exist_ok=True)
                os.rename(oldpathname, newpathname)
            for pathname in L[lastfile:]:
                parsed_filename = pathname.split('/')
                parsed_filename[0] = cwd
                oldpathname = '/'.join(parsed_filename)
                parsed_filename.insert(2, 'test')
                newpathname = '/'.join(parsed_filename)
                os.makedirs('/'.join(parsed_filename[:-1]), exist_ok=True)
                os.rename(oldpathname, newpathname)
            os.rmdir(cwd + '/Images/' + basedir[1])

if __name__ == "__main__":
    prepare_dataset()



    
