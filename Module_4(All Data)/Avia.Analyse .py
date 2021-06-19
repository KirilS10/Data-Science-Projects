#!/usr/bin/env python
# coding: utf-8

# # Task

# In this project, we need to find out which low-profit flights from Anapa can be abandoned in the winter season.

# # Importing Python Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from geopy import distance
from math import sin, cos, sqrt, radians, asin
import seaborn as sns
plt.style.use('seaborn')
import matplotlib.gridspec as gridspec


# # Loading data

# In[2]:


df=pd.read_csv('Avia_result.csv')


# # Exploratory data analysis of  data

# In[3]:


df.head(5)


# # Coloumns/features in data

# In[4]:


df.columns


# # Length of data

# In[5]:


print('Length of data is ',len(df))


# # Shape of data

# In[6]:


df.shape


# # Data information
# 

# In[7]:


df.info()


# # Additional information 

# Fuel Consumption(KG/HOUR)

# In[8]:


fuel_consumption = {'Boeing 737-300': 2600, 'Sukhoi Superjet-100':1700}


# Avg.Fuel Cost in the winter season(Russian rubles)

# In[9]:


fuel_price = {1: 41435, 2:39553, 12:47101}


# Calculation(the price of aviation fuel per flight)

# In[10]:


def price(row):
    duration= row['flight_time']/60 # получаем часы из длительности в минутах
    price_kg = row['fuel_price']/1000 # рассчитываем стоимость топлива в кг
    fuel_consumption = row['fuel_consumption'] 
    return duration*price_kg*fuel_consumption


# # Calculation(the distance between Anapa and destinations (km))

# In[33]:


Radius = 6373.0#Radius of earth(km)
def distance_cal (row):
    lon0 = 37.35
    lat0 = 45
    lat1 = row['latitude']
    lon1 = row['longitude']
    lon0, lat0, lon1, lat1 = map(radians, [lon0, lat0, lon1, lat1])
    distance_lon = lon1 - lon0
    distance_lat = lat1 - lat0

    x = sin(distance_lat / 2)**2 + cos(lat0) * cos(lat1) * sin(distance_lon / 2)**2
    y = 2 *asin(sqrt(x))

    distance_t = Radius * y

    return distance_t


df['distance_t'] = df.apply (lambda row: distance_cal(row), axis=1)


# # Converting Time

# In[12]:


df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])


# In[13]:


df['scheduled_departure_month'] = df['scheduled_departure'].apply(lambda x: x.month)


# # Duration of our flights 

# In[14]:


df['flight_time'] = df['scheduled_arrival'] - df['scheduled_departure']


# # Load Factor(Empty Seats)

# In[15]:


df['Empty seats'] = round(df.sold_seats*100/df.availableseats, 2)


# In[16]:


df[df['Empty seats'] < 75]


# # Earned amount (per minute of flight)

# In[17]:


df['flight_time'] = df['flight_time'].apply(lambda x: x/np.timedelta64(1,'m'))
df['total'] =  df.total_amount/df['flight_time']


# In[18]:


fig, ax = plt.subplots(figsize=(15, 5))

sns.boxplot(x='model',
            y='total',
            data=df.loc[df.loc[:, 'model'].isin(
                df.loc[:, 'model'].value_counts().index[:])],
            ax=ax)

plt.xticks(rotation=45)
ax.set_title('Boxplot for Aircrafts')

plt.show()


# As we can see Boeing makes more money tham Sukhoi

# # Empty Seats Proportion

# In[19]:


fig, ax = plt.subplots(figsize=(15, 5))

sns.boxplot(x='model',
            y='Empty seats',
            data=df.loc[df.loc[:, 'model'].isin(
                df.loc[:, 'model'].value_counts().index[:])],
            ax=ax)

plt.xticks(rotation=45)
ax.set_title('Boxplot for Aircrafts')

plt.show()


# In[34]:


df['fuel_consumption'] = df['model'].map(fuel_consumption)
df['fuel_price'] = df['scheduled_departure_month'].map(fuel_price)


# In[21]:


df.info()


# In[22]:


df['Cost(Flights)'] = round(df.apply(lambda x: price(x), axis=1),2)


# In[23]:


df['Profit'] = round(df['total_amount'] - df['Cost(Flights)'], 2)


# In[24]:


df['Profit(min)'] = df.Profit/df['flight_time']


# In[25]:


df.head(5)


# # Correlation between Sold tickets and Profit per min (Depending on Sold Tickets and Fuel Cost)

# In[28]:


texts = []
x1 = df['Profit(min)']
x2 = df['sold_seats']
x3 = df['flight_id']
x4= df['scheduled_departure_month']
t = df.total_amount
x5 = df['Empty seats']
fig, ax = plt.subplots()
p1 = sns.scatterplot(x=x2,
                     y=x1,
                     data=df,
                     hue='model',
                     s=100,
                     alpha=0.7,
                     edgecolor='k')
plt.show()


# # List of Flights(To Cancel )

# In[29]:


df[(df.total_amount < 1460000) & (df.model == "Boeing 737-300") | (df['Empty seats'] < 75)]


# In[32]:


df['Profit'].sum()


# In[30]:


df[(df.total_amount < 1460000) & (df.model == "Boeing 737-300") |
   (df['Empty seats'] < 75)].flight_id.to_list()

