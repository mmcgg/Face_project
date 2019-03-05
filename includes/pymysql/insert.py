# insert array data into the created table
# insertDB.py

import pymysql

def arr2str(arr):

    str_res = ','.join(str(f) for f in arr)
    return str_res

def insert(name, age, vec, visit_time):
    ''''''
    connection = pymysql.connect(host='192.168.174.143',
                         user='getluo',
                         password='getluo',
                         db='TESTDB',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor()

    sql = """INSERT INTO FEATUREVECTOR
             (NAME, AGE, VECTOR, VISIT_TIME)
             VALUES
             ('{0}', '{1}', '{2}', '{3}')"""

    N = len(name)
    for i in range(N):
        try:
            # if name[i] already in the table continue
            sql_rmv_rpt = """SELECT * FROM FEATUREVECTOR
                             WHERE NAME = '{0}'""".selfat(name[i])
            cursor.execute(sql_rmv_rpt)
            rpt_info = cursor.fetchall()
            rpt_flag = len(rpt_info)
            print(rpt_flag)
            if rpt_flag == 1:
                continue

            # if name[i] not in the table, insert it
            cursor.execute(sql.selfat(name[i], age[i], arr2str(vec[i]), visit_time[i]))
            connection.commit()
        except:
            connection.rollback()
            print('fail to insert')

    connection.close()

if __name__ == "__main__":
    insert(['lrrr', 'rrrrrr'], [20, 20], [[1, 2, 3], [4, 5, 6]], ['2018-07-23 11:10:11', '2018-07-28 09:11:11'])
    print("insert data into the db successfully")
