# -*- coding: utf-8 -*-
# createDB.py

import pymysql
from connect import *

def create():
    '''create a table in the certain database'''

    connection = connect('localhost', 'root', 'getluo', 'TESTDB')
    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS FEATUREVECTOR")
    
    sql = """CREATE TABLE FEATUREVECTOR (
             ID INT(11) NOT NULL AUTO_INCREMENT,
             NAME CHAR(30) NOT NULL COLLATE utf8_bin NOT NULL,
             AGE INT(4) NOT NULL,
             VECTOR CHAR(100) NOT NULL,
             VISIT_TIME DATETIME NOT NULL,
             PRIMARY KEY (ID)
             )ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
             AUTO_INCREMENT=1;"""

    cursor.execute(sql)

    connection.close()

if __name__ == "__main__":
    create()
    print("create data table successfully!")
