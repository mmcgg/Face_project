# -*- coding: utf-8 -*-
# delete.py

import pymysql
from connect import *

def delete(info, method=0):
    '''delete rows by the certain info'''
    connection = connect('localhost', 'root', 'getluo', 'TESTDB')
    cursor = connection.cursor()

    method_arr = ['NAME', 'AGE', 'VECTOR', 'VISIT_TIME']

    sql = """DELETE FROM FEATUREVECTOR
             WHERE
                {0} = '{1}'
            """.format(method_arr[method], info)

    try:
        cursor.execute(sql)
        connection.commit()
    except:
        connection.rollback()

    connection.close()

def delete_all():
    '''delete all the info from the table'''
    connection = connect('localhost', 'root', 'getluo', 'TESTDB')
    cursor = connection.cursor()

    sql = "DELETE FROM FEATUREVECTOR"

    try:
        cursor.execute(sql)
        connection.commit()
    except:
        connection.rollback()

    connection.close()

if __name__ == "__main__":
    # delete('luo')
    delete_all()
    print("delete successfully!")
