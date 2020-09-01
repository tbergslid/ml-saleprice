import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

def main():
    # Read input data.
    data = pd.read_csv('TrainAndValid.csv')

    # Find number of unique entries per attribute.
    attrib_uniques = pd.DataFrame(columns=['nuniques'], index=list(data))
    for column in data:
        attrib_uniques.loc[column] = data[column].nunique()

    # Drop the categories that have been determined to be of little importance.
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

    # Drop sales of "antique models" which may skew the price higher as collectibles.
    # This statement also drops rows where the YearMade does not make sense (is 1000).
    data = data[(data.YearMade >= 1955)]

    ## We remove all machines made before 1955, so giving the invalid year machines the YearMade of 1940
    ## instead should not collide with any other machines. 1000 doesn't work in the datetime methods.
    #data = data[(data.YearMade >= 1960) | (data.YearMade == 1000)]
    #data['YearMade'].replace(1000, 1940, inplace=True)

    # Fill missing attributes.

    # Hydraulics
    data['Hydraulics'].fillna(value='None or Unspecified', inplace=True)

    # auctioneerID
    # ID 1 is by far the most common, therefore we assign the entries without ID to it.
    data['auctioneerID'].fillna(value=1, inplace=True)

    # Track_Type
    # Give the non-tracked machines a Track_Type value of 'None'.
    data.update(data.loc[(data['ProductGroup'] == 'BL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'WL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'SSL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'MG'), 'Track_Type'].fillna(value='None'))
    # Assume that most TTTs have steel tracks, and the same for the TEX with missing info.
    data.update(data.loc[(data['ProductGroup'] == 'TTT'), 'Track_Type'].fillna(value='Steel'))
    data.update(data.loc[(data['ProductGroup'] == 'TEX'), 'Track_Type'].fillna(value='Steel'))

    # Hypothesis: saledate is not very interesting in itself. What is more interesting is the time elapsed
    # between YearMade and saledate. Calculate this here and save as an integer number of months.
    # Also save number of years.
    data['YearMade'] = pd.to_datetime(data['YearMade'], format="%Y")
    data['saledate'] = pd.to_datetime(data['saledate'], format="%m/%d/%Y %H:%M")
    data['ageAtSaletime'] = data['saledate'] - data['YearMade']
    # Convert YearMade back to simple int.
    data['YearMade'] = pd.DatetimeIndex(data['YearMade']).year
    # Simply divide by the average number of days/month and days/year. Should be good enough.
    data['ageAtSaletimeInMonths'] = round(data['ageAtSaletime'].dt.days/30.44).astype(int)
    data['ageAtSaletimeInYears'] = round(data['ageAtSaletime'].dt.days/365.25).astype(int)

    # Remove rows with negative age. That is, entries where saledate comes before YearMade.
    # It is likely that the two columns have simply been swapped around, but as it only
    # affects 15 entries we might as well remove them, since we can't know for sure.
    data = data[(data.ageAtSaletimeInMonths > 0)]

    # Remove 0s from MachineHoursCurrentMeter, as they can not be trusted to actually represent zero hours used.
    # We see that after this replace operation, UsageBand and MachineHoursCurrentMeter
    # are missing values in the exact same locations, as one would expect.
    data['MachineHoursCurrentMeter'].replace(['0', 0], np.nan, inplace=True)
    # Replace the NaN values with 'Unknown'.
    data['MachineHoursCurrentMeter'].fillna(value='Unknown', inplace=True)
    data['UsageBand'].fillna(value='Unknown', inplace=True)

    # Add "None or Unspecified" tag to the few Enclosure entries where that is missing.
    data['Enclosure'].fillna(value='None or Unspecified', inplace=True)
    # Assume that "EROPS AC" is the same as "EROPS w AC".
    data['Enclosure'].replace('EROPS AC', 'EROPS w AC', inplace=True)
    data['Enclosure'].replace('NO ROPS', 'None or Unspecified', inplace=True)

    # Remove rows where the price is a big statistical outlier for machines built that year.
    yearStats = data.groupby('YearMade')['SalePrice'].describe()
    newData = pd.DataFrame(columns=list(data))
    for year in yearStats.index.tolist():
        newData = newData.append(data[(data['YearMade'] == year) &
                                      (data['SalePrice'] < (yearStats.loc[year, 'mean'] +
                                                            2.0*yearStats.loc[year, 'std'])) &
                                      (data['SalePrice'] > (yearStats.loc[year, 'mean'] -
                                                            2.0*yearStats.loc[year, 'std']))])
    data = newData.copy()

#    # Rather than removing outlier prices based on YearMade, look at the ageAtSaletimeInYears.
#    yearStats = data.groupby('ageAtSaletimeInYears')['SalePrice'].describe()
#    newData = pd.DataFrame(columns=list(data))
#    for year in yearStats.index.tolist():
#        newData = newData.append(data[(data['ageAtSaletimeInYears'] == year) &
#                                      (data['SalePrice'] < (yearStats.loc[year, 'mean'] +
#                                                            2.0*yearStats.loc[year, 'std'])) &
#                                      (data['SalePrice'] > (yearStats.loc[year, 'mean'] -
#                                                            2.0*yearStats.loc[year, 'std']))])
#    data = newData.copy()

    # Check percentage of null values per attribute after cleaning data.
    null_values = data.isnull().sum()
    null_values = pd.DataFrame(null_values, columns=['null'])
    null_percent = round(null_values['null']/len(data)*100, 3)
    null_values['percent'] = null_percent
    null_values = null_values.sort_values('percent', ascending=False)
    print(null_values)

    # Plot a heatmap of the relationship between two features.
    plt.hexbin(data['YearMade'], data['SalePrice'], gridsize=50, bins='log', cmap=plt.cm.Greens)
    #plt.hexbin(data['ageAtSaletimeInYears'], data['SalePrice'], gridsize=50, bins='log', cmap=plt.cm.Greens)
    plt.colorbar(label='log10(N)')
    plt.show()

    # Print cleaned data to new csv file.
    data.to_csv('TrainAndValid_clean.csv', index=False)


if __name__ == '__main__':
    main()
