import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from functools import partial
import io
import os
from sqlalchemy import *
import re

'''
wt >> weights (number of repetitions in string) given to each feature (productName, ingredientList, productCategory, and productType)
label >> name of the column you are using as label
Returns: A dataframe of labels and corresponding text features (a string)
'''
def getFeatureString(wt, label):
    # establish connection
    metadata = MetaData()
    engine = create_engine(
        'mysql+pymysql://gc:compare123@gcapp.c4xzfsrbmzt9.us-east-1.rds.amazonaws.com:3306/app_backend_db')
    connection = engine.connect()  
    # load the product table and nutrition table
    table = Table(
        "ProductTableMerged",
        metadata,
        autoload=True,
        autoload_with=engine)

    #select feature columns of interest -  CHANGE HERE TO CHANGE TRAINING FEATURES
    s = select([table.columns.productName, table.columns.ingredientList,
                table.columns.productCategory, table.columns.productType])
    columns = pd.read_sql(s, connection)
    columns = columns.sample(n=100, random_state=1)

    #tuple list of products in form (label, feature string)
    labels = []
    features = []

    #apply function constructs tuple
    def getTuple(row, wt, label):
        #obtain feature strings, multiplying by weight to create repetitions
        try:
            name = wt[0] * (row['productName'] + " ")
        except BaseException:
            name = ""
        try:
            ingr = wt[1] * (row['ingredientList'] + " ")
        except BaseException:
            ingr = ""
        try:
            cat = wt[2] * (row['productCategory'] + " ")
        except BaseException:
            cat = ""
        try:
            typ = wt[3] * (row['productType'] + " ")
        except BaseException:
            typ = ""

        label = row[label]
        label = cleanLabel(label)
        labels.append(label)
        prod_str = name + ingr + cat + typ
        features.append(prod_str)

    columns.apply(lambda row: getTuple(row, wt, label), axis=1)

    df = pd.DataFrame()
    df['labels'] = labels
    df['text'] = features

    return df

'''
Confirms that the label is valid (checks for manual entry label errors, like spelling, extra spacing)
label >> name of the label
is_cat >> booleon indicating if labels are categories (else, they are types)
'''
'''
Confirms that the label is valid (checks for manual entry label errors, like spelling, extra spacing)
label >> name of the label
is_cat >> booleon indicating if labels are categories (else, they are types)
'''
def cleanLabel(label, is_cat = True):
    label = (str(label)).lower()
    
    #valid categories/type names 
    cats = ['produce', 'dairy & eggs', 'frozen foods', 'beverages', 
            'snacks, chips, salsas & dips','pantry', 'breads & bakery', 
            'meat', 'seafood', 'prepared food']
    typs = ['fresh fruit', 'fresh vegetables', 'pre-cut & ready to eat', 'butter & margarine', 'eggs & egg substitutes',
           'milks', 'cream', 'yogurt, pudding & jello', 'cheese', 'frozen breakfast', 'frozen entrees & appetizers', 'frozen fruits & vegetables',
           'frozen doughs','ice cream & frozen desserts', 'coffee, tea, & kombucha', 'juice', 'soft drinks', 'sports, energy, & nutritional drinks',
           'cocktail & drink mixes', 'water, seltzer, & sparkling water', 'candy & chocolate', 'chips', 'cookies & crackers,', 'miscellaneous snacks',
           'nutrition & granola bars', 'nuts, seeds & dried fruit', 'salsas, dips & spreads', 'baking', 'breakfast', 'canned & preserved goods',
           'jam, jellies & nut butters', 'rice, pasta, beans & grain', 'seasoning, sauces, condiments & dressings', 'breads', 'pastries & desserts', 
           'tortillas & flat breads', 'deli meat', 'hot dogs, bacon & sausage', 'meat alternatives', 'poultry', 'beef','pork', 'veal, game & specialty',
           'fish', 'shellfish', 'prepared meals', 'prepared sides']
    
    #RE of valid categories
    c_re = ['.*produce.*', '.*dairy.*', '.*frozen.*food.*',
             '.*beverage.*', '.*snack.*chip.*salsa.*dip.*', 
             '.*pantry.*', '.*bread.*bakery.*','.*meat.*',
             '.*seafood.*', '.*prepared.*food']
    #RE of valid types
    t_re = ['.*fresh.*fruit.*', '.*fresh.*vegetable.*', '.*pre.*cut.*.*ready.*eat.*', '.*butter.*margarine.*', '.*egg.*substitutes.*',
           '.*milk.*', '.*cream.*', '.*yogurt.*pudding.*jello.*', '.*cheese.*', '.*frozen.*breakfast.*', '.*frozen.*entree.*appetizer.*', '.*frozen.*fruit.*vegetable.*',
           '.*frozen.*doughs.*','.*frozen.*ice cream.*frozen.*dessert.*', '.*coffee.*tea.*kombucha.*', '.*juice.*', '.*soft.*drink.*', '.*sport.*energy.*nutritional.*drink.*',
           '.*cocktail.*drink.*mixes.*', '.*water.*seltzer.*sparkling.*', '.*candy.*chocolate.*', '.*chips.*', '.*cookies.*cracker.*', '.*miscellaneous.*snack.*',
           'nutrition & granola bars', 'nuts, seeds & dried fruit', 'salsas, dips & spreads', 'baking', 'breakfast', 'canned & preserved goods',
           '.*jam.*jellies.*nut.*butter.*', '.*rice.*pasta.*beans.*grain.*', '.*seasoning.*sauce.*condiment.*dressing.*', '.*bread.*', '.*pastries.*desserts.*', 
           '.*tortillas.*flat breads.*', 'deli meat', 'hot dogs, bacon & sausage', 'meat alternatives', 'poultry', 'beef','pork', 'veal, game & specialty',
           '.*fish.*', '.*shellfish.*', '.*prepared.*meals.*', '.*prepared.*sides.*']
    
    if is_cat: #cleaning categories 
        for i in range(len(c_re)): 
            newlabel = re.sub(c_re[i], cats[i], label)

            #stop if correct label is found
            if newlabel != label:
                label = newlabel
                break
    else: #cleaning types
        for i in range(len(t_re)): 
            newlabel = re.sub(t_re[i], typs[i], label)

            #stop if correct label is found
            if newlabel != label:
                label = newlabel
                break

    return label
    
'''
Returns: new merged dataframe 
Purpose: Creates a new table in the DB that has all the retagged products with all desired features 
csv >> path to csv file that contains retagged data
new_tablename >> name of the new table to generate 
source_table >> name of the table in the db with which to merge
'''
def combineRetagged(csv, new_tablename, source_table):
    #TODO 

    retag = pd.read_csv(csv) 
    
    #establish connection
    metadata=MetaData()
    engine=create_engine(source_table)
    connection=engine.connect()#load the product table and nutrition table
    table=Table("ProductTableMerged", metadata, autoload=True, autoload_with=engine)
    s = select([table.columns.productCode, table.columns.ingredientList, 
                table.columns.productDescription, table.columns.productNameSearch])
    
    columns = pd.read_sql(s,connection)
    
    merged = pd.merge(retag, columns, on='productCode')
    
    merged.to_sql(new_tablename, con=engine, if_exists='replace')

    return merged
    
    
    




def main():
    #example usage
    prods = getFeatureString([3,1,3,3], "productCategory")
    print(prods[0:10])

    print(cleanLabel("adadada produceAHH"))

if __name__ == '__main__':
    main()
