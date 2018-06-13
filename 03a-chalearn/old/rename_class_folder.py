import os
import glob

# current folder
print("Current directory", os.getcwd())

# read all subfolder names
liFolders = glob.glob("*")
print("Subfolders:", liFolders)

# loop through folders
for sFolder in liFolders:
    nClass = int(sFolder)
    sClass = "c{:03d}".format(nClass)
    print("Rename folder {} to {}".format(sFolder, sClass))
    os.rename(sFolder, sClass)