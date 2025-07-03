
import pandas as pd
import pymysql

# Establish connection
conn = pymysql.connect(
    host="localhost",       # or your host name/IP address
    user="root",   # your MySQL username
    password= "1010", # your MySQL password
    database="world" # your database name
)


db = pd.read_sql("select * from city", conn)
print(db)


# Create a cursor object to execute queries
# cursor = db.cursor()

# Execute a query
# cursor.execute("SELECT * FROM city")

# Fetch results
# results = cursor.fetchall()
# for row in results:
#     print(row)

# Close the connection when done
# cursor.close()
# db.close()