# -*- coding: utf-8 -*-
# search.py

import pymysql
from connect import *

def str2arr(str):

    arr = str.split(',')
    # convert char to float
    farr = [float(c) for c in arr]
    return farr

def search(info, method=0):
    '''
    search all the satisfied data according to the name
    while mthod=0 search data by name ;
    while method=1 search data by age;
    while method=2 search data by vector;
    while method=3 search data by datetime;
    '''
    method_arr = ['NAME', 'AGE', 'VECTOR', 'DATETIME']

    connection = connect('localhost', 'root', 'getluo', 'TESTDB')
    cursor = connection.cursor()


    sql = """SELECT NAME, AGE, VECTOR, VISIT_TIME FROM FEATUREVECTOR
             WHERE {0} = '{1}'
             ORDER BY VISIT_TIME DESC
             LIMIT 0, 2000""".format(method_arr[method], info)
    # print(sql)

    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for i, row in enumerate(results):
            name = row['NAME']
            age = row['AGE']
            vec = row['VECTOR']
            visit_time = row['VISIT_TIME']
            msg = "{0}. name: {1}, age: {2}, vec: [{3}], visit_time: {4}"
            print(msg.format(i, name, age, vec, visit_time))
    except:
        print("Error: unable to fetch data")

    # format the returned results [{}, {}] -> [[], []]
    results_format = []
    tmp = []
    for i in range(len(results)):
        tmp.append(results[i]['NAME'])
        tmp.append(results[i]['AGE'])
        tmp.append(results[i]['VECTOR'])
        tmp.append(results[i]['VISIT_TIME'])
        results_format.append(tmp)
        tmp = []

    # print(str(results_format[0][3]))
    return results_format

if __name__ == "__main__":
    search('1,2,3', 2)
    print("search successfully!")
