import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

def main():
    # Read input data.
    #data = pd.read_csv('TrainMini.csv')
    data = pd.read_csv('TrainAndValid.csv')
    #data.info()

    # Find number of unique entries per attribute.
    attrib_uniques = pd.DataFrame(columns=['nuniques'], index=list(data))
    #attrib_uniques.set_index(list(data))
    for column in data:
        attrib_uniques.loc[column] = data[column].nunique()

    # Find percentage of null values per attribute.
    print('Finding percentage of missing values for each column:')
    null_values = data.isnull().sum()
    null_values = pd.DataFrame(null_values, columns=['null'])
    null_percent = round(null_values['null']/len(data)*100, 3)
    null_values['percent'] = null_percent
    null_values = null_values.sort_values('percent', ascending=False)
    #print(null_values)

    # Sort the list of nuniques according to the percentage of null values
    attrib_uniques = attrib_uniques.reindex(null_values.index)
    #print(attrib_uniques)
    # We see that the attributes with most missing values have very few unique values.

    # 21 of the features have more than 80% of the values missing. Let us look closer at those.
    # Examine the data for attributes with lots of missing values.
    #print('Unique entries per attribute:')
    #for attrib in null_values.index[30:]:
    #    print(attrib + ':')
    #    print(data[attrib].value_counts())


    # Compare unique fiModelDesc with unique fiBaseModel. Maybe using
    # fiBaseModel as a feature provides enough detail.
    #print('Unique fiModelDesc: ', data['fiModelDesc'].nunique())
    #print('Unique fiBaseModel: ', data['fiBaseModel'].nunique())

    # Compare average price of different fiModelDesc with the general fiBaseModel they belong to
    #models = ['310G', '310SE', '310E', '310D']
    #print('310: ', data[data.fiBaseModel == '310'].SalePrice.mean())
    #for model in models:
    #    print(model, data[data.fiModelDesc == model].SalePrice.mean())

    data = data.drop(columns=['SalesID'])
    data = data.drop(columns=['Forks'])
    data = data.drop(columns=['datasource'])
    data = data.drop(columns=['Coupler'])
    data = data.drop(columns=['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
                              'fiModelDescriptor', 'ProductSize', 'ProductGroupDesc', 'Drive_System',
                              'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
                              'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Pushblock', 'Ripper',
                              'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler_System', 'Grouser_Tracks',
                              'Hydraulics_Flow', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
                              'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
                              'Differential_Type', 'Steering_Controls'], axis=1)

    # # # # ##################### TEMPORARY
    #data = data.drop(columns=['UsageBand', 'MachineHoursCurrentMeter'])
    print(data.groupby('UsageBand')['MachineHoursCurrentMeter'].mean())

    #data['Forks'].fillna(value='None or Unspecified', inplace=True)
    #data['Coupler'].fillna(value='None or Unspecified', inplace=True)

    # # # # ##################### TEMPORARY

    #data.info()

    #print(len(data))
    #antiqueData = data[(data.YearMade > 1900) & (data.YearMade < 1945)]
    # Drop "antique models" which may skew the price higher as collectibles.
    # This statement also drops rows where the YearMade does not make sense (is 1000).
    #data.info()
    data = data[(data.YearMade >= 1960)]
    #data = data[(data.YearMade >= 1960) | (data.YearMade == 1000)]
    # We remove all machines made before 1945, so giving the invalid year machines the YearMade of 1940
    # instead should not collide with any other machines. 1000 doesn't work in the datetime methods.
    #data['YearMade'].replace(1000, 1940, inplace=True)
    #data.info()
    #print(len(data))
    #print(len(antiqueData))
    #print(antiqueData)
    #print(antiqueData['MachineID'].nunique())

    # Find percentage of null values per attribute.
    null_values = data.isnull().sum()
    null_values = pd.DataFrame(null_values, columns=['null'])
    null_percent = round(null_values['null']/len(data)*100, 3)
    null_values['percent'] = null_percent
    null_values = null_values.sort_values('percent', ascending=False)
    #print(null_values)

    # Fill missing attributes.
    # Hydraulics
    #print('Hydraulics: ', data.Hydraulics.value_counts())
    #print(data.Hydraulics.head(20))
    #data.iloc[data.Hydraulics.isnull()].fillna('None or Unspecified')
    #print(data['Hydraulics'].isnull().sum())
    data['Hydraulics'].fillna(value='None or Unspecified', inplace=True)
    #print('Hydraulics: ', data.Hydraulics.value_counts())
    #print(data.loc[(data['ProductGroup'] != 'BL')]['Hydraulics'].isnull().sum())

    # auctioneerID
    # ID 1 is by far the most common, therefore we assign the entries without ID to it.
    data['auctioneerID'].fillna(value=1, inplace=True)
    #print('auctioneerID: ', data.auctioneerID.value_counts())

    # Track_Type
    #print('Track_Type')
    #print(data['Track_Type'].value_counts())
    #print(data.loc[(data['ProductGroup'] == 'TEX')]['Track_Type'].isnull().sum())
    #print(data.loc[(data['ProductGroup'] == 'BL')]['Track_Type'].isnull().sum())
    #print(data.loc[(data['ProductGroup'] == 'TTT')]['Track_Type'].isnull().sum())
    #print(data.loc[(data['ProductGroup'] == 'WL')]['Track_Type'].isnull().sum())
    #print(data.loc[(data['ProductGroup'] == 'SSL')]['Track_Type'].isnull().sum())
    #print(data.loc[(data['ProductGroup'] == 'MG')]['Track_Type'].isnull().sum())

    # Give the non-tracked machines a Track_Type value of 'None'.
    data.update(data.loc[(data['ProductGroup'] == 'BL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'WL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'SSL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'MG'), 'Track_Type'].fillna(value='None'))
    # Assume that most TTTs have steel tracks, and the same for the TEX with missing info.
    data.update(data.loc[(data['ProductGroup'] == 'TTT'), 'Track_Type'].fillna(value='Steel'))
    data.update(data.loc[(data['ProductGroup'] == 'TEX'), 'Track_Type'].fillna(value='Steel'))

#    data.loc[(data['ProductGroup'] == 'BL'), 'Track_Type'] = \
#        data.loc[(data['ProductGroup'] == 'BL'), 'Track_Type'].fillna(value='None')
#    data.loc[(data['ProductGroup'] == 'WL'), 'Track_Type'] = \
#        data.loc[(data['ProductGroup'] == 'WL'), 'Track_Type'].fillna(value='None')
#    data.loc[(data['ProductGroup'] == 'SSL'), 'Track_Type'] = \
#        data.loc[(data['ProductGroup'] == 'SSL'), 'Track_Type'].fillna(value='None')
#    data.loc[(data['ProductGroup'] == 'MG'), 'Track_Type'] = \
#        data.loc[(data['ProductGroup'] == 'MG'), 'Track_Type'].fillna(value='None')
    #data.loc[(data['ProductGroup'] == 'WL')]['Track_Type'].fillna(value='None', inplace=True)
    #data.loc[(data['ProductGroup'] == 'SSL')]['Track_Type'].fillna(value='None', inplace=True)
    #data.loc[(data['ProductGroup'] == 'MG')]['Track_Type'].fillna(value='None', inplace=True)
    #data['Track_Type'].fillna(value='None')
    #print(data.loc[(data['ProductGroup'] == 'TEX')]['Track_Type'].isnull().sum())
    #print(data['Track_Type'].value_counts())

    # saledate is not very interesting in itself. What is more interesting is the time elapsed
    # between YearMade and saledate. Calculate this here and save as an integer number of months.
    # Also save number of years.
    # Assume that machines were built in the middle of the year. Should even out.
    #data['YearMade'] = pd.to_datetime(data['YearMade'], format="%Y") + timedelta(180)
    data['YearMade'] = pd.to_datetime(data['YearMade'], format="%Y")
    data['saledate'] = pd.to_datetime(data['saledate'], format="%m/%d/%Y %H:%M")
    data['ageAtSaletime'] = data['saledate'] - data['YearMade']
    # Convert YearMade back to simple int.
    data['YearMade'] = pd.DatetimeIndex(data['YearMade']).year
    # Simply divide by the average number of days/month and days/year. Should be good enough.
    data['ageAtSaletimeInMonths'] = round(data['ageAtSaletime'].dt.days/30.44).astype(int)
    data['ageAtSaletimeInYears'] = round(data['ageAtSaletime'].dt.days/365.25).astype(int)
    #print(data[['SalePrice', 'MachineID', 'ModelID', 'YearMade', 'saledate',
    #            'ageAtSaletime', 'ageAtSaletimeInMonths', 'ageAtSaletimeInYears']].head(25))

    # Remove rows with negative age. That is, entries where saledate comes before YearMade.
    # It is likely that the two columns have simply been swapped around, but as it only
    # affects 15 entries we might as well remove them.
    data = data[(data.ageAtSaletimeInMonths > 0)]

    # Remove 0s from MachineHoursCurrentMeter, as they do not actually represent zero hours used.
    # We see that after this replace operation, UsageBand and MachineHoursCurrentMeter
    # are missing values in the exact same locations, as one would expect.
    data['MachineHoursCurrentMeter'].replace(['0', 0], np.nan, inplace=True)
    # Replace the NaN values with 'Unknown'.
    data['MachineHoursCurrentMeter'].fillna(value='Unknown', inplace=True)
    data['UsageBand'].fillna(value='Unknown', inplace=True)

#    print('Average hours in use per month for UsageBands:')
#    print(data.loc[(data['UsageBand'] == 'Low')]['MachineHoursCurrentMeter'].divide(
#        data.loc[(data['UsageBand'] == 'Low')]['ageAtSaletimeInMonths']).mean())
#    print(data.loc[(data['UsageBand'] == 'Medium')]['MachineHoursCurrentMeter'].divide(
#        data.loc[(data['UsageBand'] == 'Medium')]['ageAtSaletimeInMonths']).mean())
#    print(data.loc[(data['UsageBand'] == 'High')]['MachineHoursCurrentMeter'].divide(
#          data.loc[(data['UsageBand'] == 'High')]['ageAtSaletimeInMonths']).mean())

    # Add "None or Unspecified" tag to the few Enclosure entries where that is missing.
    data['Enclosure'].fillna(value='None or Unspecified', inplace=True)
    #data.update(data.loc[(data['Enclosure'] == 'EROPS AC')].fillna(value='None'))
    # Assume that "EROPS AC" is the same as "EROPS w AC".
    data['Enclosure'].replace('EROPS AC', 'EROPS w AC', inplace=True)
    data['Enclosure'].replace('NO ROPS', 'None or Unspecified', inplace=True)

    # Remove rows where the price is a big statistical outlier for that year.
    # print(data.groupby(['YearMade', 'SalePrice']).describe())
    # yearStats = data[data.duplicated(['YearMade'], keep=False)]
    # yearStats = pd.DataFrame(columns=list(np.unique(data['YearMade'])))
    # print(data.groupby(['YearMade', 'SalePrice']).sum().reset_index().groupby('YearMade')['SalePrice'].mean())
    yearStats = data.groupby('YearMade')['SalePrice'].describe()
    #print(yearStats.loc[1950, 'std'])
    print(yearStats.index.tolist())
    #print(yearStats['std'])
    # for year in yearStats.index.tolist():
    # year = 1952
    newData = pd.DataFrame(columns=list(data))
    newRows = []
    print(newData)
    for year in yearStats.index.tolist():
        #print(year)
        #print(yearStats.loc[year, 'mean'])
        #print(yearStats.loc[year, 'std'])
        newData = newData.append(data[(data['YearMade'] == year) &
                                      (data['SalePrice'] < (yearStats.loc[year, 'mean'] +
                                                            2.0*yearStats.loc[year, 'std'])) &
                                      (data['SalePrice'] > (yearStats.loc[year, 'mean'] -
                                                            2.0*yearStats.loc[year, 'std']))])
        #newData.append(newRows)
        #year = 1952
        #newData = data[(data['YearMade'] == year) &
    #               (data['SalePrice'] < (yearStats.loc[year, 'mean'] + 2.0 * yearStats.loc[year, 'std']))]
    #newData = pd.concat(newRows)
    print(newData)
    data = newData.copy()
    # data = data[(data.SalePrice < yearStats)]
    # for year in np.unique(data['YearMade']):
    # for year in yearStats:
    #    print(yearStats[year])
    # yearStats[year] = data.loc[(data.YearMade == year), 'SalePrice'].tolist()
    # print(data.groupby('YearMade').filter(lambda x:  x year).mean())
    # data.groupby(['YearMade', 'SalePrice']).describe()

    # print(yearStats[1950].mean())
    # print(yearStats[1950].min())
    # print(yearStats[1950].max())
    # Check percentage of null values per attribute after cleaning data.
    null_values = data.isnull().sum()
    null_values = pd.DataFrame(null_values, columns=['null'])
    null_percent = round(null_values['null']/len(data)*100, 3)
    null_values['percent'] = null_percent
    null_values = null_values.sort_values('percent', ascending=False)
    #print(null_values)

#    ageSorted = data.sort_values(by='ageAtSaletimeInYears')
#    print(ageSorted[['MachineID', 'SalePrice', 'YearMade', 'saledate',
#                     'ageAtSaletime', 'ageAtSaletimeInMonths', 'ageAtSaletimeInYears']].head(50))

    #data.info()
    #print(data.head(20))
    #print(data['Track_Type'].isnull().sum())
    #print(data['Enclosure'].value_counts())

    #print(data['SalePrice'][0:100])
    #print(data['SalePrice'].describe())
    #print('Cheapest machine: ', min(data['SalePrice']))
    #print('Most expensive machine: ', max(data['SalePrice']))
    #print('Average cost:', sum(data['SalePrice'])/len(data['SalePrice']))

    #inputVariables = "'MachineID', 'ModelID', 'YearMade', 'MachineHoursCurrentMeter','fiModelDesc', 'ProductGroup'"
    #inputVariables = {'MachineID', 'ModelID', 'YearMade', 'MachineHoursCurrentMeter',
    #                  'fiModelDesc', 'ProductGroup'}
    #print(inputVariables)

    #sns.set_style('whitegrid')

    #pricePlot = data.groupby(['SalesID'])['SalePrice'].sum().reset_index()
    #modelPlot = data.groupby(['SalesID'])['fiModelDesc'].sum().reset_index()

    #pricePlot['SalePrice'].describe()
    #pricePlot['SalePrice'].plot(kind='hist')
    #modelPlot['fiModelDesc'].apply(pd.value_counts).plot(kind='bar')

    #print(np.where(pd.isnull(data['Pushblock']))[0])
    #print(len(np.where(pd.isnull(data['Pushblock']))[0]))

    #print(dictionary['Variable'][1])

    #print(data['MachineID'].value_counts())
    #print(data['MachineID'].nunique())

    # Check how many machines have been sold more than nSold times.
    #n_sold = 20
    #print('Machines sold multiple times: ')
    #filterMachineID = np.where(data['MachineID'].value_counts() > n_sold)[0]

    #print(data['MachineID'].iloc[filterMachineID])
    #print('Unique fiModelDesc: ', data['fiModelDesc'].nunique())
    #print('Unique ModelID: ', data['ModelID'].nunique())

    #mostSold = data[data['MachineID'].nunique() > 5]
    #print(mostSold)

#    for key in list(data):
#        print('Unique ', key, ': ', data[key].nunique())
#        # Count number of sales with no info for a given key.
#        print('Zero: ', len(np.where(pd.isnull(data[key]))[0]))

    #print(data['ModelID'].apply(pd.value_counts))
    #print(modelPlot['fiModelDesc'].apply(pd.value_counts))
    #plt.show()

    #filterYear = np.where(data['YearMade'] > 1800)[0]
    #print(filterYear)
    #plt.scatter(data['YearMade'].iloc[filterYear], data['SalePrice'].iloc[filterYear])
    plt.scatter(data['YearMade'], data['SalePrice'])
    #plt.scatter(data['Hydraulics'], data['SalePrice'])
    #plt.scatter(data['auctioneerID'], data['SalePrice'])
    #plt.scatter(data['Track_Type'], data['SalePrice'])
    #plt.scatter(data['saledate'], data['SalePrice'])
    #plt.scatter(data['Enclosure'], data['SalePrice'])
    #plt.xlabel("X")
    #plt.ylabel("SalePrice")
    plt.show()

#    #sns.pairplot(data[['SalePrice', 'MachineID', 'ModelID', 'ageAtSaletimeInYears']], hue='SalePrice')
#    #sns.pairplot(data[['MachineID', 'ModelID', 'ageAtSaletimeInYears']], hue='ProductGroup')
#    sns.set(style='ticks', color_codes=True)
#    #sns.pairplot(data[['SalePrice', 'ModelID', 'ageAtSaletimeInYears']], hue='ProductGroup')
#    print(data['ProductGroup'].value_counts())
#    data.info()
#    #sns.pairplot(data, vars=data[['SalePrice', 'YearMade', 'ageAtSaletimeInYears']], hue='ProductGroup')
#    sns.pairplot(data, vars=data[['SalePrice', 'ModelID', 'ageAtSaletimeInYears']])
#    plt.show()


    # Print cleaned data to new csv file.
    data.to_csv('TrainAndValid_clean.csv', index=False)

    #print(len(data))
    #print(np.where(data['YearMade'] > 1950))
    #print(data[data['YearMade'] > 1950]['YearMade'])


if __name__ == '__main__':
    main()
