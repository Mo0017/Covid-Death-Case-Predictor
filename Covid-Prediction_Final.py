#Importing libraries needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import datetime as dt
from datetime import timedelta
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from pylab import *
import plotly.express as px
from prophet import Prophet
pd.options.mode.chained_assignment = None  # default='warn'
#Initializing Data
cases = pd.read_csv('CodeWithMosh\AI4ALL\Covid-Cases.csv')
vaccs = pd.read_csv('CodeWithMosh\AI4ALL\State-Vacs.csv')

#Checking for Null values
print(cases.isnull().sum())
print(vaccs.isnull().sum())

#Cleaning data, drops extra columns, fixes date format, fills in emtpy spaces
cases.drop(["Unnamed: 2"], axis = 1, inplace = True)
cases = cases.replace(np.nan,0)
vaccs['people_vaccinated'] = vaccs["people_vaccinated"].fillna(0).astype("int")
vaccs['people_fully_vaccinated'] = vaccs["people_fully_vaccinated"].fillna(0).astype("int")
# print("Cases shape: ",cases.shape, "Vaccine shape: ",vaccs.shape)
cases['date'] = pd.to_datetime(cases['date'])
vaccs['date'] = pd.to_datetime(vaccs['date'])

#Check if Null
print(cases.isnull().sum())
print(vaccs.isnull().sum())

#Info on the data. 
total_cases = cases.index[cases['date'] == "2022-11-21"]
print("Total cases and deaths by each selected state as of 2022-11-21")
print(cases.iloc[total_cases].to_string(index = False))

#Individual data for each state. copying the data from the cases file and vaccine file and creating a new data 
# set for each state. Also cleaning the data, filling in empty spaces, adding a new row for daily cases, calculating it.

#Alabama
alabama_data = cases.query('state == "Alabama"').reset_index(drop = True)
alabama_data["daily cases"] = alabama_data.cases.diff()
if(math.isnan(alabama_data.at[0,"daily cases"])):
     alabama_data.at[0,"daily cases"] = alabama_data.at[0,"cases"]
alabama_data["daily cases"] = alabama_data["daily cases"].astype('int') 
alabama_data_vaccs = vaccs.query('location == "Alabama"').reset_index(drop= True)
alabama_data_vaccs["people_vaccinated"] = alabama_data_vaccs["people_vaccinated"].astype('int')
alabama_data['people_vaccinated'] = 0
alab_vacc_index = int(alabama_data.index.values[alabama_data['date'] == '2021-01-12'])
alab_series = pd.Series(alabama_data_vaccs['people_vaccinated']) 
alabama_data['people_vaccinated'] = pd.Series(data = alab_series.values, index = alabama_data.index[alabama_data.index >= alab_vacc_index].to_series().iloc[:alab_series.size])
alabama_data['people_vaccinated'] = alabama_data["people_vaccinated"].fillna(0).astype("int")
monthly_alabama_data = alabama_data[alab_vacc_index:-1]
monthly_alabama_data = monthly_alabama_data.filter(monthly_alabama_data['date'].dt.to_period('M').drop_duplicates().index, axis = 0)
#print(monthly_alabama_data.to_string()

#Mississippi 
mississippi_data = (cases.query('state == "Mississippi"').reset_index(drop = True))
mississippi_data["daily cases"] = mississippi_data.cases.diff()
if(math.isnan(mississippi_data.at[0,"daily cases"])):
    mississippi_data.at[0,"daily cases"] = mississippi_data.at[0,"cases"]
mississippi_data["daily cases"] = mississippi_data["daily cases"].astype('int')
mississippi_data_vaccs = vaccs.query('location == "Mississippi"').reset_index(drop= True)
mississippi_data_vaccs["people_vaccinated"] = mississippi_data_vaccs["people_vaccinated"].astype('int')
mississippi_data['people_vaccinated'] = 0
miss_vacc_index = int(mississippi_data.index.values[mississippi_data['date'] == '2021-01-12'])
miss_series = pd.Series(mississippi_data_vaccs['people_vaccinated']) 
mississippi_data['people_vaccinated'] = pd.Series(data = miss_series.values, index = mississippi_data.index[mississippi_data.index >= miss_vacc_index].to_series().iloc[:miss_series.size])
mississippi_data['people_vaccinated'] = mississippi_data["people_vaccinated"].fillna(0).astype("int")
monthly_mississippi_data = mississippi_data[miss_vacc_index:-1]
monthly_mississippi_data = monthly_mississippi_data.filter(monthly_mississippi_data['date'].dt.to_period('M').drop_duplicates().index, axis = 0)
#print(monthly_mississippi_data)

#Wyoming 
wyoming_data = (cases.query('state == "Wyoming"').reset_index(drop = True))
wyoming_data["daily cases"] = wyoming_data.cases.diff()
if(math.isnan(wyoming_data.at[0,"daily cases"])):
    wyoming_data.at[0,"daily cases"] = wyoming_data.at[0,"cases"]
wyoming_data["daily cases"] = wyoming_data["daily cases"].astype('int')
wyoming_data_vaccs = vaccs.query('location == "Wyoming"').reset_index(drop= True)
wyoming_data_vaccs["people_vaccinated"] = wyoming_data_vaccs["people_vaccinated"].astype('int')
wyoming_data['people_vaccinated'] = 0
wyo_vacc_index = int(wyoming_data.index.values[wyoming_data['date'] == '2021-01-12'])
wyo_series = pd.Series(wyoming_data_vaccs['people_vaccinated']) 
wyoming_data['people_vaccinated'] = pd.Series(data = wyo_series.values, index = wyoming_data.index[wyoming_data.index >= wyo_vacc_index].to_series().iloc[:wyo_series.size])
wyoming_data['people_vaccinated'] = wyoming_data["people_vaccinated"].fillna(0).astype("int")
wyoming_data["daily cases"] = wyoming_data["daily cases"].astype('int')
monthly_wyoming_data = wyoming_data[wyo_vacc_index:-1]
monthly_wyoming_data = monthly_wyoming_data.filter(monthly_wyoming_data['date'].dt.to_period('M').drop_duplicates().index, axis = 0)
#print(monthly_wyoming_data)

#Rhode Island 
rhode_island_data = (cases.query('state == "Rhode Island"').reset_index(drop = True))
rhode_island_data["daily cases"] = rhode_island_data.cases.diff()
rhode_island_data['people_vaccinated'] = 0
if(math.isnan(rhode_island_data.at[0,"daily cases"])):
    rhode_island_data.at[0,"daily cases"] = rhode_island_data.at[0,"cases"]
rhode_island_data["daily cases"] = rhode_island_data["daily cases"].astype('int')
rhode_island_data_vaccs = vaccs.query('location == "Rhode Island"').reset_index(drop= True)
rhode_island_data_vaccs["people_vaccinated"] = rhode_island_data_vaccs["people_vaccinated"].astype('int')
rho_vacc_index = int(rhode_island_data.index.values[rhode_island_data['date'] == '2021-01-12'])
rho_series = pd.Series(rhode_island_data_vaccs['people_vaccinated']) 
rhode_island_data['people_vaccinated'] = pd.Series(data = rho_series.values, index = rhode_island_data.index[rhode_island_data.index >= rho_vacc_index].to_series().iloc[:rho_series.size])
rhode_island_data['people_vaccinated'] = rhode_island_data["people_vaccinated"].fillna(0).astype("int")
monthly_rhode_island_data = rhode_island_data[rho_vacc_index:-1]
monthly_rhode_island_data = monthly_rhode_island_data.filter(monthly_rhode_island_data['date'].dt.to_period('M').drop_duplicates().index, axis = 0)
#print(monthly_rhode_island_data)


#Vermont 
vermont_data = (cases.query('state == "Vermont"').reset_index(drop = True))
vermont_data["daily cases"] = vermont_data.cases.diff()
if(math.isnan(vermont_data.at[0,"daily cases"])):
    vermont_data.at[0,"daily cases"] = vermont_data.at[0,"cases"]
vermont_data["daily cases"] = vermont_data["daily cases"].astype('int')
vermont_data_vaccs = vaccs.query('location == "Vermont"').reset_index(drop= True)
vermont_data_vaccs["people_vaccinated"] = vermont_data_vaccs["people_vaccinated"].astype('int')
ver_vacc_index = int(vermont_data.index.values[vermont_data['date'] == '2021-01-14'])
ver_series = pd.Series(vermont_data_vaccs['people_vaccinated']) 
vermont_data['people_vaccinated'] = pd.Series(data = ver_series.values, index = vermont_data.index[vermont_data.index >= ver_vacc_index].to_series().iloc[:ver_series.size]-2)
vermont_data['people_vaccinated'] = vermont_data["people_vaccinated"].fillna(0).astype("int")
monthly_vermont_data = vermont_data[ver_vacc_index: -1]
monthly_vermont_data = monthly_vermont_data.filter(monthly_vermont_data['date'].dt.to_period('M').drop_duplicates().index, axis = 0)
#print(monthly_vermont_data)

# Massachusets 
massachusetts_data = (cases.query('state == "Massachusetts"').reset_index(drop = True))
massachusetts_data["daily cases"] = massachusetts_data.cases.diff()
if(math.isnan(massachusetts_data.at[0,"daily cases"])):
     massachusetts_data.at[0,"daily cases"] = massachusetts_data.at[0,"cases"]
massachusetts_data["daily cases"] = massachusetts_data["daily cases"].astype('int')
massachusetts_data_vaccs = vaccs.query('location == "Massachusetts"').reset_index(drop= True)
massachusetts_data_vaccs["people_vaccinated"] = massachusetts_data_vaccs["people_vaccinated"].astype('int')
mass_vacc_index = int(massachusetts_data.index.values[massachusetts_data['date'] == '2021-01-12'])
mas_series = pd.Series(massachusetts_data_vaccs['people_vaccinated']) 
massachusetts_data['people_vaccinated'] = pd.Series(data = mas_series.values, index = massachusetts_data.index[massachusetts_data.index >= mass_vacc_index].to_series().iloc[:ver_series.size])
massachusetts_data['people_vaccinated'] = massachusetts_data["people_vaccinated"].fillna(0).astype("int")
monthly_massachusetts_data = massachusetts_data[mass_vacc_index:-1]
monthly_massachusetts_data = monthly_massachusetts_data.filter(monthly_massachusetts_data['date'].dt.to_period('M').drop_duplicates().index, axis = 0)
#print(monthly_massachusetts_data)


 #Graph of cases, highest cases as well as total vaccinations 
states = ['Rhode Island', 'Vermont', 'Massachusetts','Alabama', 'Wyoming', 'Mississippi']
case = [rhode_island_data['cases'].max(),vermont_data['cases'].max(),massachusetts_data['cases'].max(),alabama_data['cases'].max(),
wyoming_data['cases'].max(),mississippi_data['cases'].max()]
x = np.arange(len(states))
width = 0.6
fig, ax = plt.subplots()
bar1 = ax.bar(x, case, width, color = ['royalblue', 'forestgreen', 'teal', 'darkred','sienna', 'brown'] , label = "Cases")
ax.set_ylabel('Total Cases')
ax.set_xlabel("States")
ax.set_title('Number of Cases per State')
ax.set_xticks(x, states)
ax.bar_label(bar1, fmt = '%d')
plt.ticklabel_format(style='plain', axis = "y")
plt.show()

#Bar graph of cases
vacc = [rhode_island_data_vaccs['people_vaccinated'].max(),vermont_data_vaccs['people_vaccinated'].max(),massachusetts_data_vaccs['people_vaccinated'].max(),alabama_data_vaccs['people_vaccinated'].max(),
wyoming_data_vaccs['people_vaccinated'].max(),mississippi_data_vaccs['people_vaccinated'].max()]
x = np.arange(len(states))
width = 0.6
fig, ax = plt.subplots()
bar1 = ax.bar(x, vacc, width, color = ['royalblue', 'forestgreen', 'teal', 'darkred','sienna', 'brown'] , label = "Vaccs")
ax.set_ylabel('Total Vaccs')
ax.set_xlabel("States")
ax.set_title('Highest and Lowest Vaccinated States')
ax.set_xticks(x, states)
ax.bar_label(bar1, fmt = '%d')
plt.ticklabel_format(style='plain', axis = "y")
plt.show()

#Line graph of vaccinations
sns.lineplot(data = monthly_rhode_island_data,x = 'date', y = 'people_vaccinated', color = 'royalblue', label = 'Rhode Island')
sns.lineplot(data = monthly_vermont_data,x ='date',y = 'people_vaccinated', color = 'forestgreen',label = 'Vermont')
sns.lineplot(data = monthly_massachusetts_data,x ='date',y = 'people_vaccinated',color = 'teal',label = 'Massachusetts')
sns.lineplot(data = monthly_alabama_data,x ='date',y = 'people_vaccinated', color = 'darkred',label = 'Alabama')
sns.lineplot(data = monthly_wyoming_data,x ='date',y = 'people_vaccinated', color = 'sienna',label = 'Wyoming')
sns.lineplot(data = monthly_mississippi_data,x ='date',y = 'people_vaccinated', color = 'brown',label = 'Mississippi')
plt.legend()
plt.ticklabel_format(style='plain', axis = "y")
plt.xticks(rotation=45, horizontalalignment='right',)
plt.title('Timeline of Vaccinations For Each State', fontsize = 12)
plt.xlabel("Date")
plt.ylabel("Vaccines Administered")
plt.show()

# #Daily cases
rhode_island_graph = px.bar(rhode_island_data,x = 'date', y = 'daily cases', title = "Rhode island Daily Cases",color_discrete_sequence =['royalblue']*3)
rhode_island_graph.show()

vermont_graph = px.bar(vermont_data,x = 'date', y = 'daily cases', title = "Vermont Daily Cases",color_discrete_sequence =['forestgreen']*3)
vermont_graph.show()

massachusetts_graph = px.bar(massachusetts_data,x = 'date', y = 'daily cases', title = "Massachusetts Daily Cases",color_discrete_sequence =['teal']*3)
massachusetts_graph.show()

alabama_graph = px.bar(alabama_data,x = 'date', y = 'daily cases', title = "Alabama Daily Cases",color_discrete_sequence =['darkred']*3)
alabama_graph.show()

wyoming_graph = px.bar(wyoming_data,x = 'date', y = 'daily cases', title = "Wyoming Daily Cases",color_discrete_sequence =['sienna']*3)
wyoming_graph.show()

mississippi_graph = px.bar(mississippi_data,x = 'date', y = 'daily cases', title = "Mississippi Daily Cases",color_discrete_sequence =['brown']*3)
mississippi_graph.show()

#Data Training
all_states = pd.concat([rhode_island_data,vermont_data,massachusetts_data,alabama_data,wyoming_data,mississippi_data],ignore_index = True)
print(all_states.head())
print(all_states.isnull().sum())
print(all_states.shape)

trained_states = all_states.groupby(['date','state']).agg('sum').reset_index()
print(trained_states.tail(6))

#Compare 2 states cases, (Top 3 highest vaccinated vs lowest 3 vaccinated respectively)
def cases_compare(time, *argv):
    Coun1 = argv[0]
    Coun2=argv[1]
    fig, ax = plt.subplots(figsize=(16,5))
    labels=argv  
    highest_state=trained_states.loc[(trained_states['state'] == Coun1)]
    plt.plot(highest_state['date'],highest_state['cases'],linewidth=2)
    plt.legend(labels)
    ax.set(title = ' Timeline of Confirmed Cases',ylabel='Number of cases' )

    lowest_state=trained_states.loc[trained_states['state']==Coun2]
    lowest_state['date']=lowest_state['date']-datetime.timedelta(days=time)
    plt.plot(lowest_state['date'],lowest_state['cases'],linewidth=2)
    plt.legend(labels)
    ax.set(title = ' Evolution of cases in %d days difference '%time ,ylabel='Number of Cases')
    plt.ticklabel_format(style='plain', axis = "y")
    plt.show()

#print(cases_compare(7, 'Rhode Island', 'Alabama'), cases_compare(7,'Vermont', 'Wyoming'), cases_compare(7, 'Massachusetts', 'Mississippi'))

#Compare 2 states deaths, (Top 3 highest vaccinated vs lowest 3 vaccinated respectively)
def deaths_compare2(time, *argv):
    Coun1 = argv[0]
    Coun2=argv[1]
    fig, ax = plt.subplots(figsize=(16,5))
    labels=argv  
    highest_state=trained_states.loc[(trained_states['state'] == Coun1)]
    plt.plot(highest_state['date'],highest_state['deaths'],linewidth=2)
    plt.legend(labels)
    ax.set(title = ' Timeline of Confirmed Deaths',ylabel='Number of Deaths' )

    lowest_state=trained_states.loc[trained_states['state']==Coun2]
    lowest_state['date']=lowest_state['date']-datetime.timedelta(days=time)
    plt.plot(lowest_state['date'],lowest_state['deaths'],linewidth=2)
    plt.legend(labels)
    ax.set(title = ' Evolution of Deaths in %d days difference '%time ,ylabel='Number of deaths')
    plt.ticklabel_format(style='plain', axis = "y")
    plt.show()

print(deaths_compare2(7, 'Rhode Island', 'Alabama'), deaths_compare2(7,'Vermont', 'Wyoming'), deaths_compare2(7, 'Massachusetts', 'Mississippi'))
print(all_states.info())

#Forecasting data, to see prediction of other states swap out where marked
trained_states = all_states.loc[:,['date', 'deaths']] # <-- swap deaths for cases 
trained_states['date'] = pd.DatetimeIndex(trained_states['date'])
print(trained_states.dtypes)

def state__deaths_prediction(trained_states):
    trained_states = trained_states.rename(columns={'date': 'ds','deaths': 'y'})

    my_model = Prophet()
    my_model.fit(trained_states)

 
    future_dates = my_model.make_future_dataframe(periods=100)
    forecast =my_model.predict(future_dates)

    fig2 = my_model.plot_components(forecast)


    forecastnew = forecast['ds']
    forecastnew2 = forecast['yhat']

    forecastnew = pd.concat([forecastnew,forecastnew2], axis=1)
    mask = (forecastnew['ds'] > "11-21-2022") & (forecastnew['ds'] <= "12-25-2022")
    forecastedvalues = forecastnew.loc[mask]

    mask = (forecastnew['ds'] > "8-12-2022") & (forecastnew['ds'] <= "11-20-2022")
    forecastnew = forecastnew.loc[mask]
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(forecastnew.set_index('ds'), color='b')
    ax1.plot(forecastedvalues.set_index('ds'), color='r')
    ax1.set_ylabel('cases')
    ax1.set_xlabel('date')
    plt.title('Massachusetts Deaths Prediction')
    plt.ticklabel_format(style='plain', axis = "y")
    plt.show()
    print("Red = Predicted Values, Blue = Base Values")

trained_states_state = all_states[all_states['state']== 'Massachusetts']

state__deaths_prediction(trained_states_state)
