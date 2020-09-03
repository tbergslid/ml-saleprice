import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # Read input data.
    data = pd.read_csv('TrainAndValid.csv')
    data.info()

    # Plot a histogram of the price distribution.
    plt.hist(data['SalePrice'], bins=np.linspace(data['SalePrice'].min(), data['SalePrice'].max()))
    plt.axis([0, 150000, 0, 40000])
    plt.xlabel('SalePrice')
    plt.ylabel('Transactions')
    plt.title('Sales')
    plt.grid()
    plt.show()
    plt.cla()

    # Plot a histogram of the production years.
    # The year 1000 is still in the data at this point, which messes up the bin distribution.
    # Quick and dirty fix is to just use enough bins to span all years from 1000-2014.
    plt.hist(data['YearMade'], bins=1015)
    plt.axis([1915, 2014, 0, 23000])
    plt.xlabel('YearMade')
    plt.ylabel('Number of machines')
    plt.title('Production')
    plt.grid()
    plt.show()
    plt.cla()

    # Calculate statistics for interesting variables.
    print('{: <24} {: <14} {: <14} {: <14}'.format('', 'mean', 'min', 'max'))
    print('{: <24} {: <14.2f} {: <14.2f} {: <14.2f}'.format('SalePrice',
                                                            data['SalePrice'].mean(),
                                                            data['SalePrice'].min(),
                                                            data['SalePrice'].max()))
    print('{: <24} {: <14.2f} {: <14.2f} {: <14.2f}'.format('YearMade',
                                                            data['YearMade'].mean(),
                                                            data['YearMade'].min(),
                                                            data['YearMade'].max()))
    print('{: <24} {: <14.2f} {: <14.2f} {: <14.2f}'.format('MachineHoursCurrentMeter',
                                                            data['MachineHoursCurrentMeter'].mean(),
                                                            data['MachineHoursCurrentMeter'].min(),
                                                            data['MachineHoursCurrentMeter'].max()))

    # Print mean hours in each UsageBand.
    print(data.groupby('UsageBand')['MachineHoursCurrentMeter'].mean())

    # Plot SalePrice statistics by year.
    years = sorted(set(data['YearMade']))
    # Leave the year 1000 out of this...
    years = years[1:]
    year_mean = data.groupby('YearMade')['SalePrice'].mean()
    year_max = data.groupby('YearMade')['SalePrice'].max()
    year_min = data.groupby('YearMade')['SalePrice'].min()
    year_mean = year_mean[1:]
    year_max = year_max[1:]
    year_min = year_min[1:]
    plt.scatter(years, year_mean, marker='o', color='blue', label='mean')
    plt.scatter(years, year_max, marker='x', color='green', label='max')
    plt.scatter(years, year_min, marker='x', color='red', label='min')
    # Connect the max and min markers with a line.
    plt.plot((years, years), ([i for i in year_min], [j for j in year_max]), color='black')
    plt.axis([years[0]-1, years[-1]+1, 0, data['SalePrice'].max()])
    plt.grid()
    plt.legend()
    plt.title('Price statistics by year')
    plt.show()
    plt.cla()

    # Find percentage of null values per attribute.
    print('Finding percentage of missing values for each column:')
    null_values = data.isnull().sum()
    null_values = pd.DataFrame(null_values, columns=['null'])
    null_percent = round(null_values['null']/len(data)*100, 3)
    null_values['percent'] = null_percent
    null_values = null_values.sort_values('percent', ascending=False)
    print(null_values)

    # Find number of unique entries per attribute.
    attrib_uniques = pd.DataFrame(columns=['nuniques'], index=list(data))
    for column in data:
        attrib_uniques.loc[column] = data[column].nunique()
    # Sort the list of nuniques according to the percentage of null values.
    attrib_uniques = attrib_uniques.reindex(null_values.index)
    print(attrib_uniques)
    # We see that the attributes with most missing values have very few unique values.

    # 21 of the features have more than 80% of the values missing. Let us look closer at those.
    print('Unique entries for each attribute:')
    for attrib in null_values.index[:21]:
        print(attrib + ':')
        print(data[attrib].value_counts())

    # Compare unique fiModelDesc with unique fiBaseModel. Maybe using
    # fiBaseModel as a feature provides enough detail.
    #print('Unique fiModelDesc: ', data['fiModelDesc'].nunique())
    #print('Unique fiBaseModel: ', data['fiBaseModel'].nunique())

    # Compare average price of different fiModelDesc with the general fiBaseModel they belong to
    #models = ['310G', '310SE', '310E', '310D']
    #print('310: ', data[data.fiBaseModel == '310'].SalePrice.mean())
    #for model in models:
    #    print(model, data[data.fiModelDesc == model].SalePrice.mean())
    # Pretty big difference on the first arbitrarily checked models, so probably not worth pursuing.

    # Drop features we don't want to examine further from the dataset.
    data = data.drop(columns=['SalesID'])
    data = data.drop(columns=['Forks'])
    data = data.drop(columns=['datasource'])
    data = data.drop(columns=['Coupler'])
    data = data.drop(columns=['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor',
                              'ProductSize', 'ProductGroupDesc', 'Drive_System',
                              'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
                              'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Pushblock', 'Ripper',
                              'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler_System', 'Grouser_Tracks',
                              'Hydraulics_Flow', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
                              'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
                              'Differential_Type', 'Steering_Controls'], axis=1)

    data.info()

    # Check how many machines have been sold more than nSold times.
    n_sold = 20
    print('Machines sold more than {} times: '.format(n_sold))
    filterMachineID = np.where(data['MachineID'].value_counts() > n_sold)[0]
    print('{} machines sold more than {} times.'.format(len(filterMachineID), n_sold))
    if len(filterMachineID) < 50:
        print(data['MachineID'].iloc[filterMachineID])

    # Now let us drop the rows with YearMade==1000, so we can look at some plots by year.
    data = data[(data.YearMade >= 1900)]

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

    # Plot a heatmap of price distributions by YearMade.
    plt.hexbin(data['YearMade'], data['SalePrice'], gridsize=50, bins='log', cmap=plt.cm.Greens)
    plt.axis([data['YearMade'].min(), data['YearMade'].max(),
              data['SalePrice'].min(), data['SalePrice'].max()])
    plt.colorbar(label='N')
    plt.xlabel('YearMade')
    plt.ylabel('SalePrice')
    plt.title('SalePrice distribution')
    plt.show()

    # Use pairplots to visualize relationships between features.
    sns.set(style='ticks', color_codes=True)
    sns.pairplot(data, vars=data[['SalePrice', 'ModelID', 'ageAtSaletimeInYears']])
    plt.show()

