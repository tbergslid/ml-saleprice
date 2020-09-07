import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # Read input data.
    data = pd.read_csv('TrainAndValid.csv')

    # Drop the features that have been determined to be of little importance.
    # The features on separate lines are the features that looked like they might have some impact based
    # on the number of missing values and unique entries. Easier to swap them back in this way.
    data = data.drop(columns=['datasource'])
    data = data.drop(columns=['Forks'])
    data = data.drop(columns=['Coupler'])
    data = data.drop(columns=['Grouser_Type'])
    data = data.drop(columns=['Blade_Type'])
    data = data.drop(columns=['Blade_Width'])
    data = data.drop(columns=['Scarifier'])
    data = data.drop(columns=['Steering_Controls'])
    data = data.drop(columns=['Travel_Controls'])
    data = data.drop(columns=['Thumb'])
    data = data.drop(columns=['Ride_Control'])
    data = data.drop(columns=['Differential_Type'])
    data = data.drop(columns=['SalesID', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries',
                              'fiModelDescriptor', 'ProductGroupDesc',
                              'Pad_Type', 'Stick', 'Turbocharged', 'Blade_Extension',
                              'Enclosure_Type', 'Engine_Horsepower', 'Pushblock', 'Ripper',
                              'Tip_Control', 'Tire_Size', 'Coupler_System', 'Grouser_Tracks',
                              'Hydraulics_Flow', 'Undercarriage_Pad_Width', 'Stick_Length',
                              'Pattern_Changer', 'Backhoe_Mounting',
                              ], axis=1)

    # Drop sales of "antique models" which may skew the price higher as collectibles.
    # This statement also drops rows where the YearMade does not make sense (is 1000).
    data = data[(data.YearMade >= 1959)]
    #data = data[(data.YearMade <= 2008)]

    # Alternative method of dealing with all the machines with YearMade == 1000:
    ## We remove all machines made before 1960, so giving the invalid year machines the YearMade of 1940
    ## instead should not collide with any other machines. 1000 doesn't work in the datetime methods
    ## and this will also shorten the range between min and max values.
    #data = data[(data.YearMade >= 1960) | (data.YearMade == 1000)]
    #data['YearMade'].replace(1000, 1940, inplace=True)

#    # We have very little data for transactions with SalePrice over 100,000.
#    # Could be beneficial to drop them?
#    data = data[(data.SalePrice < 100000)]

    # Fill missing attributes.

    # Fill in missing values for features that we suspect may have an impact.
    # Use the value already present in the dataset that would make sense, or "Unknown" if not.
    data['Hydraulics'].fillna(value='None or Unspecified', inplace=True)
    data['ProductSize'].fillna(value='Unknown', inplace=True)
    data['Drive_System'].fillna(value='Unknown', inplace=True)
    data['Transmission'].fillna(value='None or Unspecified', inplace=True)

    # ID 1 is by far the most common, therefore we assign the entries without ID to it.
    data['auctioneerID'].fillna(value=1, inplace=True)

    # Give the non-tracked machines a Track_Type value of 'None'.
    data.update(data.loc[(data['ProductGroup'] == 'BL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'WL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'SSL'), 'Track_Type'].fillna(value='None'))
    data.update(data.loc[(data['ProductGroup'] == 'MG'), 'Track_Type'].fillna(value='None'))
    # TTTs are mostly missing Track_Type info. Steel is most likely, but let us call it Unknown.
    data.update(data.loc[(data['ProductGroup'] == 'TTT'), 'Track_Type'].fillna(value='Unknown'))
    # Steel tracks are most common, so we assign Steel to the few TEX machines with empty Track_Type fields.
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

    # Check variable correlation
    #plt.cla()
    #sns.heatmap(data[['SalePrice', 'YearMade', 'ageAtSaletimeInYears']].corr(), annot=True, cmap=plt.cm.Reds)
    #plt.show()

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

