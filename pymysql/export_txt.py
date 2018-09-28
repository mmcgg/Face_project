# -*- coding: utf-8 -*-
# export_txt.py

import pymysql
from connect import *

def export(abst_path):
    '''export the sql file to a txt file'''

    connection = connect('localhost', 'root', 'getluo', 'TESTDB')
    cursor = connection.cursor()

    sql = """SELECT * INTO OUTFILE '/home/txtfile/FEATUREVECTOR.txt'
             FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
             LINES TERMINATED BY '\n'
             FROM FEATUREVECTOR"""

    try:
        cursor.execute(sql)
    except:
        print('can not export to a txt file')

if __name__ == "__main__":
    export(1)
