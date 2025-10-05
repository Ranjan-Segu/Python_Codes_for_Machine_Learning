
                             ####PANDAS_FUNCTIONS####


#Importing

import pandas as pd


#Series

s = pd.Series([3, -5, 7, 4], index=["a", "b", "c", "d"])

  
#DataFrame

data =  {"Country": ["Belgium", "India", "Brazil"], "Capital": ["Brussels", "New Delhi", "Brasilia"], "Population": [11190846, 1303171035, 207847528]}

df = pd.DataFrame(data, columns=["Country", "Capital", "Population"])


#Help

help(pd.Series.loc)


#Selection

s["b"]        

df[1:]                                                   

df.iloc[0, 0]

df.iat[0, 0]

df.loc[0, "Country"]

df.at[1, "Country"]


#Indexing

s[~(s > 1)]

s[(s < -1) | (s > 2)]

df[df["Population"] > 1200000000]


#Data_Manipulation

s["a"] = 6

s.drop(["a", "c"])

df.drop("Country", axis = 0)

df.sort_index()

df.sort_values(by="Country")

df.rank()


#DataFrame_Information

df.shape

df.index

df.columns

df.info

df.count

df.sum()

df.cumsum()

df["Population"].min()/df["Population"].max()

df.idxmin()
df.idxmax()
df.idxmin()/df.idxmax()

df.describe()
df["Population"].mean()
df["Population"].median()

f = lambda x : x*2
df.apply(f)
df.map(f)


s3 = pd.Series([7, -2, 3], index=["a", "c", "d"])

s + s3
s.add(s3, fill_value=13)
s.sub(s3, fill_value=2)
s.div(s3, fill_value=4)
s.mul(s3, fill_value=3)


#Read_&_Write

pd.read_csv("diabetes.csv", nrows=5)

df.to_csv("myDataFrame.csv")

import sys


pd.read_excel("DAXFunctions200116.xls")
xlxs = pd.ExcelFile("DAXFunctions200116.xls")

df.to_excel("myDataFrame.xlsx", sheet_name = "ranjan")
df = pd.read_excel("myDataFrame.xlsx", "ranjan")


#SQL

# =========================
# Fully self-contained Spyder script
# =========================

import pandas as pd
import sqlalchemy as sa

# 1️⃣ Connect to SQLite database
db_path = "my_database.db"  # Replace with your SQLite DB path
engine = create_engine("sqlite:///:memory:")
connection = engine.connect()
metadata = sa.MetaData()

users = sa.Table('users', metadata, sa.Column('id', sa.Integer, primary_key=True), sa.Column('name', sa.String), sa.Column('age', sa.Integer))

metadata.create_all(engine)

pd.read_sql("SELECT * FROM users;", engine)
pd.read_sql_query("SELECT * FROM users;", engine)
pd.read_sql_table("users", engine)

df.to_sql("df", engine)

