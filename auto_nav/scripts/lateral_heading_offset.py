#!/usr/bin/env python
import numpy as np
import cv2
import glob
import pyexcel as pe

records = pe.iget_records(file_name="../config/coordinates_utm.ods")
for record in records:
    print("The latitude and longitutdes are %s and %s" % (record['Lat(North)'], record['Lon(East)']))
