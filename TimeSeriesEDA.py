# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:44:54 2023

@author: seelc
"""
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



'''Part 1: Data Load and cleaning'''

airport = pd.read_csv('large sample, 16M.csv')

print(list(airport.columns))
'''Part 2: EDA'''

#Delay vs no delay
sns.countplot(data=airport, x = 'Delay')
plt.title('Airplane Delay Frequency')
plt.show()

#Now examining single variable correlation with delay
sns.countplot(data=airport,x = 'Month',  hue = 'Delay')
plt.title('Airplane Delay Frequency By Month')
plt.show()

sns.countplot(data=airport,x = 'Year',  hue = 'Delay')
plt.title('Airplane Delay Frequency By Year')
plt.show()

sns.countplot(data=airport,x = 'Quarter',  hue = 'Delay')
plt.title('Airplane Delay Frequency By Quarter')
plt.show()

sns.countplot(data=airport,x = 'DayOfWeek',  hue = 'Delay')
plt.title('Airplane Delay Frequency By DayOfWeek')
plt.show()

sns.countplot(data=airport,x = 'DayofMonth',  hue = 'Delay')
plt.title('Airplane Delay Frequency By DayOfMonth')
plt.show()

#Now generating subplot with delay vs no delay by year, month, and dayofweek
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
sns.countplot(data=airport,x = 'Year',  hue = 'Delay', ax = axs[0])
axs[0].set_title('Delay vs Year')

sns.countplot(data=airport,x = 'Month',  hue = 'Delay', ax = axs[1])
axs[1].set_title('Delay vs Month')

sns.countplot(data=airport,x = 'DayOfWeek',  hue = 'Delay',  ax = axs[2])
axs[2].set_title('Delay vs DayOfWeek')

plt.tight_layout()
plt.suptitle('Flight Delay Chance Over Time', 
             fontsize = 15, y=1.02)
plt.show()
