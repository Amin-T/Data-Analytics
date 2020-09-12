"""
@author: Amin Tavakkolnia

The goal of this project is to compute summary statistics from comma separated files that contain trading information from Deutsche Börse.

The python program is writen that extracts the necessary fields. Spark Map-Reduce is used to compute the summary statistics per month and per symbol
and then those statistics are inserted in a My SQL database.

"""

# Imports
from datetime import datetime
from functools import partial
import re
import os
import json
import numpy as np
import glob  # This import is necessary to retrieve folder structures
from pyspark import SparkContext
import mysql.connector

# Create Spark context
sc = SparkContext()

# Initialise path
datapath = "/Project/trading/*/"
productlistpath = "/Project/productlist.csv"

# Read dataset into RDD
input = sc.textFile(datapath).map(lambda line: line.split(','))

# Initialise functions and variables
index2column = {1: "MarketSegment",
                2: "UnderlyingSymbol",
                3: "UnderlyingISIN",
                4: "Currency",
                5: "SecurityType",
                12: "Date",
                13: "Time",
                14: "StartPrice",
                15: "MaxPrice",
                16: "MinPrice",
                17: "EndPrice",
                18: "NumberOfContracts",
                19: "NumberOfTrades"}


def transform_to_json(record, index2column):
    return_value = dict()
    for index, column_name in index2column.items():
        return_value[column_name] = record[index]
    return return_value


def convert_to_float(floatstr):
    try:
        return float(floatstr)
    except ValueError:
        return None


def convert_to_int(intstr):
    try:
        return int(intstr)
    except ValueError:
        return None


def convert_to_YM(datestr):
    d = datetime.strptime(datestr, '%Y-%m-%d')
    try:
        return (d.year, d.strftime('%b'))
    except ValueError:
        return None


def convert_to_H(timestr):
    try:
        return str(datetime.strptime(timestr, "%H:%M").strftime("%H"))
    except ValueError:
        return None


def convert_to_string(string):
    try:
        return string.replace("\"", "")
    except ValueError:
        return None


type_converters = {'Date': convert_to_YM,
                   'Time': convert_to_H,
                   'StartPrice': convert_to_float,
                   'MaxPrice': convert_to_float,
                   'MinPrice': convert_to_float,
                   'EndPrice': convert_to_float,
                   'NumberOfContracts': convert_to_int,
                   'NumberOfTrades': convert_to_int,
                   'MarketSegment': convert_to_string,
                   'UnderlyingSymbol': convert_to_string,
                   'Currency': convert_to_string,
                   'SecurityType': convert_to_string,
                   'UnderlyingISIN': convert_to_string
                   }


def convert_types(record, converters):
    for col_name, convert_func in converters.items():
        record[col_name] = convert_func(record[col_name])
    return record


def summary_statistics(tpl):
    id, trade_vol = tpl
    vol_array = np.array(trade_vol)
    return {'TradeID': id[0][1] + '-' + id[1] + "-" + id[2],
            'Year': id[0][0],
            'Month': id[0][1],
            'Symbol': id[1],
            'MarketSegment': id[2],
            'sum': int(np.sum(vol_array)),
            'avg': float(np.round(np.mean(vol_array), 3)),
            'std': float(np.round(np.std(vol_array), 3)),
            'min': int(np.min(vol_array)),
            'max': int(np.max(vol_array))}


# RDD to JSON and eliminate header rows
Data_json = (input.map(partial(transform_to_json, index2column=index2column))
             .filter(lambda x: x['MarketSegment'] != 'MarketSegment')
             )

# Convert data types in JSON
Data_prep = Data_json.map(partial(convert_types, converters=type_converters))

# Summary statistics about traded volume per symbol per month
trade_vol = (Data_prep.map(lambda x: ((x['Date'], x['UnderlyingSymbol'], x['MarketSegment']), x['NumberOfTrades']))
             .groupByKey()
             .map(lambda x: (x[0], list(x[1])))
             .map(summary_statistics)
             )

with open(productlistpath, "r", encoding='utf-8', errors='ignore') as products:
    sectorlist = dict()
    for line in products:
        line_split = line.strip().split(";")
        sectorlist[line_split[0]] = line_split[14]  # 1st column has market segment, 3th column has product name

del sectorlist['PRODUCT_ID']  # Header was also imported and should be deleted

broadcast_sectorlist = sc.broadcast(sectorlist)  # Broadcast the dictionary


def add_sector(record):
    record["Sector"] = broadcast_sectorlist.value.get(record["MarketSegment"], None)  # If no match found its a None
    return record


trade_vol_full = trade_vol.map(add_sector)  # Add the sector to the rdd
trade_vol_full.take(5)


# MySQL connection function
def connection_factory(user, password, host, database):
    cnx = mysql.connector.connect(
        user=user,
        password=password,
        host=host,
        database=database
    )
    cursor = cnx.cursor()
    return cnx, cursor


connection_factory = partial(connection_factory,
                             user='root',
                             password='At@2357453',
                             host='127.0.0.1',
                             database='project_trading'
                             )


# Function to store records in a mysql table
def store_records(records, connection_factory):
    try:
        cnx, cursor = connection_factory()

        insert_statement_str = \
            "insert into TradeVolume (Year, Month, Sector, Symbol, MarketSegment, TotalVol, Mean, StDev, Min, Max) " \
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

        record_list = list()
        for record in records:
            record_list.append((
                record['Year'],
                record['Month'],
                record['Sector'],
                record['Symbol'],
                record['MarketSegment'],
                record['sum'],
                record['avg'],
                record['std'],
                record['min'],
                record['max']
            ))

        cursor.executemany(insert_statement_str, record_list)

        cnx.commit()
    finally:
        cnx.close()


# Insert data into MySql database
trade_vol_full.foreachPartition(partial(store_records, connection_factory=connection_factory))
