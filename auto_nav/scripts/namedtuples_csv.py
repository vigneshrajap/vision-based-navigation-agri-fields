## CSV file read and write with named tuples
from collections import namedtuple
import csv

def cast_if_number(s):
    try:
        return float(s)
    except ValueError:
        return s

def write_namedtuples_to_csv(filename, nt_list):
    assert all([type(a)==type(nt_list[0]) for a in nt_list]) 

    with open(filename, 'w') as csvfile:
        wr = csv.writer(csvfile, delimiter= '\t')
        wr.writerow(nt_list[0]._fields)
        for nt in nt_list:
            wr.writerow(nt)

def read_namedtuples_from_csv(filename, type_name):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        field_names = reader.next() #extract header
        nt_class = namedtuple(type_name,field_names)
        nt_list = []
        for row in reader:
            row = [cast_if_number(s) for s in row]
            nt_list.append(nt_class._make(row))
        return nt_list   