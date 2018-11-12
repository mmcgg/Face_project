import pymysql
from PyMySQL import *



py = PyMySQL('localhost', 'root', 'Asd980517', 'WEININGFACE')
py.create_table('FEATUREVECTOR')
py.insert(['e', 'f'], [20, 20], [[1, 2, 3], [4, 5, 6]], ['2018-07-23 11:10:11', '2018-07-28 09:11:11'])
py.show_all()
print('==========')
py.show_all()
res = py.search(20, 1)
print('res', res)
print('=============')
print('delete 7-23')
py.delete(20, 1)
py.show_all()
res = py.search('r')
print(res)
print(1)