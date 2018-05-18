import os
import glob

from subprocess import call, check_output

print(os.getcwd())

sVideoPath = "datasets/04-chalearn/train/001/M_00001.avi"
fVideoSec = int(check_output(["mediainfo", '--Inform=Video;%Duration%', sVideoPath]))/1000.0
print("%s: %.3f sec" % (sVideoPath, fVideoSec))