# model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 0. Load Data

data = pd.read_csv('../data/data.csv')
data.info()

# 1. Data Analysis: First Sight

# How is the date distributed?

flights_by_airline = data['OPERA'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.9)
plt.title('Flights by Airline')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Airline', fontsize=12)
plt.xticks(rotation=90)
plt.show()

flights_by_day = data['DIA'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_day.index, flights_by_day.values, color='lightblue', alpha=0.8)
plt.title('Flights by Day')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Day', fontsize=12)
plt.xticks(rotation=90)
plt.show()

flights_by_month = data['MES'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_month.index, flights_by_month.values, color='lightblue', alpha=0.8)
plt.title('Flights by Month')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation=90)
plt.show()

flights_by_day_in_week = data['DIANOM'].value_counts()
days = [
    flights_by_day_in_week.index[2],
    flights_by_day_in_week.index[5],
    flights_by_day_in_week.index[4],
    flights_by_day_in_week.index[1],
    flights_by_day_in_week.index[0],
    flights_by_day_in_week.index[6],
    flights_by_day_in_week.index[3]
]
values_by_day = [
    flights_by_day_in_week.values[2],
    flights_by_day_in_week.values[5],
    flights_by_day_in_week.values[4],
    flights_by_day_in_week.values[1],
    flights_by_day_in_week.values[0],
    flights_by_day_in_week.values[6],
    flights_by_day_in_week.values[3]
]
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(days, values_by_day, color='lightblue', alpha=0.8)
plt.title('Flights by Day in Week')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Day in Week', fontsize=12)
plt.xticks(rotation=90)
plt.show()

flights_by_type = data['TIPOVUELO'].value_counts()
sns.set(style="darkgrid")
plt.figure(figsize=(10, 2))
sns.barplot(flights_by_type.index, flights_by_type.values, alpha=0.9)
plt.title('Flights by Type')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Type', fontsize=12)
plt.show()

flight_by_destination = data['SIGLADES'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flight_by_destination.index, flight_by_destination.values, color='lightblue', alpha=0.8)
plt.title('Flight by Destination')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Destination', fontsize=12)
plt.xticks(rotation=90)

# 2. Features Generation

# 2.a. Period of Day

from datetime import datetime

def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'

data['PERIODODIA'] = data['HORA'].apply(get_period_day)
data.info()

# 2.b. Date Times

data['FECHA'] = pd.to_datetime(data['FECHA'])
data['ANO'] = data['FECHA'].dt.year
data['MES'] = data['FECHA'].dt.month
data['DIA'] = data['FECHA'].dt.day
data['HORA'] = data['FECHA'].dt.time
data['SEMANA'] = data['FECHA'].dt.isocalendar().week
data.info()

# 3. Correlations between features

sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.2, fmt=".2f")
plt.show()

# 4. Classification

# 4.a. Classification by airline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['CODAIRLINE'] = encoder.fit_transform(data['OPERA'])
X = data[['MES', 'DIA', 'SEMANA', 'PERIODODIA']]
y = data['CODAIRLINE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4.b. Classification by type of flight

data['CODTIPOVUELO'] = encoder.fit_transform(data['TIPOVUELO'])
X = data[['MES', 'DIA', 'SEMANA', 'PERIODODIA']]
y = data['CODTIPOVUELO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Time Series Analysis

plt.figure(figsize=(20, 2))
sns.set(style="darkgrid")
sns.lineplot(x='FECHA', y='VUELOS', data=data, alpha=0.9)
plt.title('Flights over Time')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.xticks(rotation=90)
plt.show()

# 6. Final Thoughts