"""
CS301-002 Introduction to Data Science
Sountongnoma Jefferson Kabore
Final project
"""

import pandas as pd
import seaborn as sns; sns.set()  # for plot styling
import matplotlib.pyplot as plt
from collections import Counter

import sklearn
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



def file_reader(a_file):
    # This reads the data set and creates an array of tuples (each tuple being an array of instances for each attribute)
    my_data = []
    my_file = open(a_file, "r")
    for line in my_file:
        line = line.strip().split(",")
        if len(line) == 8:
            my_data.append(line)
    my_file.close()
    return my_data


def doNothing():
    return 0


data = file_reader("OnlineRetail.csv")

###################################################### Question a #######################################################
histogramTable = {}  # This dictionary holds the different times and their frequencies
for element in data:
    toBeAdded = element[4][element[4].find("/201") + 6:] + ":00"
    if toBeAdded in histogramTable:
        histogramTable[toBeAdded] += 1
    elif toBeAdded == "ceDate:00":
        doNothing()
    else:
        histogramTable[toBeAdded] = 1

# This dictionary contains the formatted time values and their frequencies.
print("Question a")
print(histogramTable)
width = 1.0

# Drawing the histogram
ax = plt.axes()
plt.hist(histogramTable.values())
plt.title('Question a: Shopping time distribution per hour')
plt.xlabel('Frequencies')
plt.ylabel('Time')
plt.show()


###################################################### Question b #######################################################
coffeeCustomers = {}
data.remove(data[0])
for element in data:
    if "COFFEE" in element[2]:
        coffeeCustomers[element[6]] = float(element[5])

# Finding 10 highest values
k = Counter(coffeeCustomers)
top10 = k.most_common(10)
print("\nQuestion b")
print("The ids of the top 10 consumers that \"coffee\" related products are: ")
for each in top10:
    print(each[0])

###################################################### Question c #######################################################
bestSeller = {}
for element in data:
    bestSeller[element[2]] = int(element[3])

# Finding 10 highest values
k = Counter(bestSeller)
best10 = k.most_common(10)
print("\nQuestion c")
print("The descriptions of the top 10 best sellers for any products are: ")
for each in best10:
    print(each[0])

###################################################### Question d #######################################################
itemsets = {}
for element in data:
    if element[2] in itemsets:
        itemsets[element[2]] += 1
    else:
        itemsets[element[2]] = 1

# Finding top 5 most frequent itemsets
k = Counter(itemsets)
top5 = k.most_common(5)
print("\nQuestion d")
print("The descriptions of the top 5 most frequent itemsets are for: ")
for each in top5:
    print(each[0])

# Drawing a histogram for the most common itemsets
plt.hist(itemsets.values())
plt.title('Question d: Distribution of the top 5 most frequent itemsets')
plt.xlabel('Itemsets')
plt.ylabel('Frequencies')
plt.show()


###################################################### Question e #######################################################
print("\nQuestion e")
dataset = pd.read_csv("OR_fixed.csv")  # Reading data
print("This is what our data set looks like: ")
print(dataset.head())

# Plotting data
plt.scatter(dataset["UnitPrice"], dataset["Quantity"])
plt.title("A look at our data set")
plt.xlabel("UnitPrice")
plt.ylabel("Quantity")
plt.show()

# Clustering
x = dataset.copy()
x_scaled = sklearn.preprocessing.scale(x)
kmeans = KMeans(5)
kmeans.fit(x_scaled)
clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x_scaled)

# Plotting clusters
plt.scatter(clusters["UnitPrice"], clusters["Quantity"], c=clusters['cluster_pred'], cmap='rainbow')
plt.title("A look at our clusters")
plt.xlabel("UnitPrice")
plt.ylabel("Quantity")
plt.show()


###################################################### Question f #######################################################
print("\nQuestion f")

df = pd.read_csv("OnlineRetail.csv")
df.head()

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

basket = (df[df['Country'] == "France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("This is a look at our first 5 association rules: \n", rules.head())

rules[(rules['lift'] >= 6) &
      (rules['confidence'] >= 0.8)]
