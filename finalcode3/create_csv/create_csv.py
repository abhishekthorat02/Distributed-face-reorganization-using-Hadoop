#!/usr/bin/env python

import sys
import os.path

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: create_csv <base_path>"
    	sys.exit(1)
    BASE_PATH_FOLDER=sys.argv[1]
    BASE_PATH="../face_crop/Cropped"+str(BASE_PATH_FOLDER)
    SEPARATOR=";"

    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
		label = subdirname.split("frame")[1]
		label = int(label)
                print "%s%s%d" % (abs_path, SEPARATOR, label)
    #print "%s%s%d" % ("/tmp/inputData/reference.jpg", SEPARATOR, -2)
