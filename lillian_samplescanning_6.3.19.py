from sqlalchemy import create_engine
from sqlalchemy import Table
from sqlalchemy import *
import pandas as pd
import numpy as np
import csv
import os
import time
import re

#establish connection
metadata=MetaData()
engine=create_engine('mysql+pymysql://gc:compare123@gcapp.c4xzfsrbmzt9.us-east-1.rds.amazonaws.com:3306/app_backend_db')
connection=engine.connect()#load the product table and nutrition table
table=Table('ProductTableMerged', metadata, autoload=True, autoload_with=engine)

s = select([table.columns.productCode, table.columns.productName, table.columns.ingredientList])

all_ingr = pd.read_sql(s,connection)

ingredients = all_ingr.sample(n=10000, random_state=1)

commodities = []

def getComod(row):
	com = []

	ing = row['ingredientList']
	name = row['productName']
	x = ""

	if len(str(ing)) == 0:
		x = name
	else:
		x = ing

	match1 = re.search('.*banana.*', str(x).lower())
	if match1:
		com.append("banana")
	match2 = re.search('.*watermelon.*', str(x).lower())
	if match2:
		com.append("watermelon")

	commodities.append(', '.join(com))

start = time.process_time()
ingredients.apply(lambda row: getComod(row), axis=1)
print("Time: " + str(time.process_time()-start))

ingredients['commoditiesList'] = commodities 
ingredients.to_sql('example_scanning_lillian_hong_06', con=engine, if_exists='replace')

# print(ingredients.columns)

# arr = np.array(commodities)
# arr_ban = arr[arr=="['banana']"]
# arr_wat = arr[arr=="['watermelon']"]
# print(len(arr_ban) + len(arr_wat))



# take random sample
# productName 