from sqlalchemy import create_engine
from sqlalchemy import Table
from sqlalchemy import *
import pandas as pd
import numpy as np
import csv
import os
import time
import re

# merge the two tables [productCode, ingredientsList], on="productCode"


def getIngredients(csv_path, engine_path):

	#Get ProductTableMerged

	metadata=MetaData()
	engine=create_engine(engine_path)
	connection=engine.connect()#load the product table and nutrition table
	table=Table('ProductTableMerged', metadata, autoload=True, autoload_with=engine)

	s = select([table.columns.productCode, table.columns.ingredientList])

	all_ingr = pd.read_sql(s,connection)

	#Get productCodes from csv (primary key)
	df = pd.read_csv(csv_path)
	print(df.columns)
	prod_codes = df['productCode'] #get all product codes 

	#merge tables 
	prod_ingr = pd.merge(prod_codes, all_ingr, how="left", on="productCode")#merge tables on productCode to obtain ingredients list for products in CSV 
	print(prod_ingr)

	df['ingredientList'] = prod_ingr['ingredientList']

	df['text'] = df['text'] + " " + df['ingredientList']

	df.to_csv('sample_prod_cat_ingredients.csv')

engine_path = 'mysql+pymysql://gc:compare123@gcapp.c4xzfsrbmzt9.us-east-1.rds.amazonaws.com:3306/app_backend_db'
csv_path = 'sample_prod_cat.csv'
getIngredients(csv_path, engine_path)
print("HERE")
