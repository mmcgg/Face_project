# -*- coding: utf-8 -*-
# PyMySQL Operations

import pymysql

class PyMySQL:
	'''basic operation for pymysql'''

	def __init__(self, host, user, passwd, db):
		'''initialization'''
		self.host = host
		self.user = user
		self.passwd = passwd
		self.db = db
		self.table_name = 'FEATURE'

	def connect(self, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor):
		'''connect to a certain database'''

		connection = pymysql.connect(host=self.host,
									 user=self.user,
									 passwd=self.passwd,
									 db=self.db,
									 charset=charset,
									 cursorclass=cursorclass)
		return connection

	def create_table(self, table_name):
		'''create a table in the certain database'''

		self.table_name = table_name
		connection = self.connect()
		cursor = connection.cursor()

		cursor.execute("DROP TABLE IF EXISTS {0}".format(self.table_name))

		sql = """CREATE TABLE {0} (
				 ID INT(11) NOT NULL AUTO_INCREMENT,
				 NAME CHAR(30) NOT NULL COLLATE utf8_bin,
				 AGE INT(4) NOT NULL,
				 VECTOR CHAR(100) NOT NULL,
				 VISIT_TIME DATETIME NOT NULL,
				 PRIMARY KEY (ID)
				 ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE utf8_BIN
				 AUTO_INCREMENT=1;""".format(self.table_name)

		cursor.execute(sql)

		connection.close()

	def arr2str(self, arr):
		'''convert a arr into string format'''
		str_res = ','.join(str(f) for f in arr)
		str_res = '[' + str_res + ']'

		return str_res

	def insert(self, name, age, vec, visit_time):
		'''insert a piece of info into the database'''

		connection = self.connect()
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
								WHERE NAME = '{0}'""".format(name[i])
				cursor.execute(sql_rmv_rpt)
				rpt_info = cursor.fetchall()
				rpt_flag = len(rpt_info)
				# print(rpt_flag)
				if rpt_flag == 1:
					continue

				# if name[i] not in the table, insert it
				cursor.execute(sql.format(name[i], age[i], self.arr2str(vec[i]), visit_time[i]))
				connection.commit()

			except:
				connection.rollback()
				print('fail to insert info')

		connection.close()

	def show_all(self):
		'''show all the infos in the table, just for test'''

		connection = self.connect()
		cursor = connection.cursor()
		sql = """SELECT * FROM FEATUREVECTOR ORDER BY NAME ASC"""

		cursor.execute(sql)
		results = cursor.fetchall()
		# print(results)
		results_format = []
		tmp = []
		for i in range(len(results)):
			tmp.append(results[i]['NAME'])
			tmp.append(results[i]['AGE'])
			tmp.append(results[i]['VECTOR'])
			tmp.append(results[i]['VISIT_TIME'])
			results_format.append(tmp)
			tmp = []
		print(results_format)

	def delete(self, info, method=0):
		'''
    	search all the satisfied data according to the name
    	while mthod=0 search data by name ;
		while method=1 search data by age;
		while method=2 search data by vector;
		while method=3 search data by datetime;
    	'''
		
		# print(info[:10])
		connection = self.connect()
		cursor = connection.cursor()

		method_arr = ['NAME', 'AGE', 'VECTOR', 'VISIT_TIME']

		if method == 0 or method == 1 or method == 2:
			sql = """DELETE FROM FEATUREVECTOR
					WHERE
						{0} = '{1}'""".format(method_arr[method], info)

			try:
				cursor.execute(sql)
				connection.commit()
			except:
				connection.rollback()

		elif method == 3:
			sql = """DELETE FROM FEATUREVECTOR
					WHERE
						{0} REGEXP '{1}'""".format(method_arr[method], '^'+info[:10])
			try:
				cursor.execute(sql)
				connection.commit()
			except:
				connection.rollback()

		else:
			print('Error: wrong method')
			return 0

		connection.close()

	def delete_all(self):
		''' delete all the infos'''

		connection = self.connect()
		cursor = connection.cursor()

		sql = """DELETE FROM FEATUREVECTOR"""

		try:
			cursor.execute(sql)
			connection.commit()
		except:
			connection.rollback()

		connection.close()

	def str2arr(self, str):
		''''''
		str = str[1: len(str)-1]
		arr = str.split(',')
		farr = [float(c) for c in arr[1: len(arr)]]
		return farr

	def search(self, info, method=0):

		method_arr = ['NAME', 'AGE', 'VECTOR', 'VISIT_TIME']

		connection = self.connect()
		cursor = connection.cursor()
		# print(info[:10])

		if method == 0 or method == 1 or method == 2:
			sql = """SELECT NAME, AGE, VECTOR, VISIT_TIME FROM FEATUREVECTOR 
					WHERE {0} = '{1}' 
					ORDER BY VISIT_TIME DESC 
					LIMIT 0, 2000""".format(method_arr[method], info)
		elif method == 3:
			sql = """SELECT NAME, AGE, VECTOR, VISIT_TIME FROM FEATUREVECTOR 
					WHERE {0} REGEXP '{1}' 
					ORDER BY VISIT_TIME DESC 
					LIMIT 0, 2000""".format(method_arr[method], '^'+info[:10])

		try:
			cursor.execute(sql)
			results = cursor.fetchall()

			for i, row in enumerate(results):
				name = row['NAME']
				age = row['AGE']
				vector = row['VECTOR']
				visit_time = row['VISIT_TIME']
				msg = "{0}. name: {1}, age: {2}, vector: {3}, visit_time: {4}"
				print(msg.format(i, name, age, vector, visit_time))
		except:
			print("Error: unable to fetch data")
			return 0

		results_format = []
		tmp = []
		for i in range(len(results)):
			tmp.append(results[i]['NAME'])
			tmp.append(results[i]['AGE'])
			tmp.append(results[i]['VECTOR'])
			tmp.append(results[i]['VISIT_TIME'])
			results_format.append(tmp)
			tmp = []

		return results_format

# if __name__ == "__main__":
# 	py = PyMySQL('localhost', 'root', 'getluo', 'TESTDB')
# 	# py.create_table('FEATUREVECTOR')
# 	py.insert(['e', 'f'], [20, 20], [[1, 2, 3], [4, 5, 6]], ['2018-07-23 11:10:11', '2018-07-28 09:11:11'])
# 	py.show_all()
# 	print('==========')
# 	# py.show_all()
# 	res = py.search(20, 1)
# 	print('res', res)
# 	# print('=============')
# 	# print('delete 7-23')
# 	py.delete(20, 1)
# 	py.show_all()
# 	# # res = py.search('r')
# 	# # print(res)
# 	# print(1)


