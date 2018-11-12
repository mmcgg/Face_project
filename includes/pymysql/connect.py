# -*- coding: utf-8 -*-
# connect.py

import pymysql

def connect(host, user, password, db,
            charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor):
    '''connect to the certain database'''

    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 db=db,
                                 charset=charset,
                                 cursorclass=cursorclass)

    return connection
