# Atul Bharati
# Multivariate Linear Regression of Global Surface Temperature

###########################################
### First Plot
###########################################

# Imports packages and classes
import LHD
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_in = LHD.load_space_data('modern_climate_time_series.dat',22)
data_org = data_in  # Creates copy of original data for second plot
data_in = np.ma.masked_where(data_in == -999,data_in)  # Masks rows with '-999'
data_in = np.ma.compress_rows(data_in)  # Deletes rows containing mask '-999'

# Extracts variables
year = data_in[:,0]  # year
temp = data_in[:,1]  # global_temperature_anomaly (C)
tsi = data_in[:,2]  # total_solar_irradiance_anomaly (W/m^2)
enso = data_in[:,3]  # enso_super_el_nino_2025
vol = data_in[:,5]  # volcanic_aer_big_eruption_2015
CO2 = data_in[:,7]  # surface_radiative_forcing_co2 (W/m^2)

nobs = len(year)  # Number of observations
nrgs = 5  # Number of regressors + 1 (1 is for the y-intercept const.)

# Creates a matrix of 1s with the dimentions defined above
regressors = np.ones((nobs,nrgs),dtype=np.float)

# Populates the matrix columns with regressors (leaving column 0 as 1s) 
regressors[:,1] = tsi 
regressors[:,2] = enso
regressors[:,3] = vol
regressors[:,4] = CO2

# Creates an ordinary least squares model
results = sm.OLS(temp,regressors)
# Extracts the constants
constants = results.fit().params
print(results.fit().summary())

# Calculates the model fit result
reg = constants[0] + constants[1]*tsi + constants[2]*enso + constants[3]*vol + constants[4]*CO2

# Plot settings
fig = plt.figure(figsize=[9,12])
ax1 = plt.subplot2grid((8,1),(0,0),rowspan=2)
ax2 = plt.subplot2grid((8,1),(3,0))
ax3 = plt.subplot2grid((8,1),(4,0))
ax4 = plt.subplot2grid((8,1),(5,0))
ax5 = plt.subplot2grid((8,1),(6,0))

ax1.set_xlim(np.amin(year),np.amax(year))
ax2.set_xlim(np.amin(year),np.amax(year))
ax3.set_xlim(np.amin(year),np.amax(year))
ax4.set_xlim(np.amin(year),np.amax(year))
ax5.set_xlim(np.amin(year),np.amax(year))

ax1.xaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])
ax3.xaxis.set_ticklabels([])
ax4.xaxis.set_ticklabels([])

ax1.plot(year,temp,label='Observed Temp Anomalies')  # Plots observed temps
ax1.plot(year,reg,label='Modeled Temp Anomalies')  # Plots modeled temps
ax2.plot(year,constants[1]*tsi,c='k',label='Solar Irradiance Impact')  # Plots TSI
ax3.plot(year,constants[2]*enso,c='b',label='ENSO Impact')  # Plots ENSO
ax4.plot(year,constants[3]*vol,c='g',label='Volcanic Aerosols Impact')  # Plots Volcanic Aerosols
ax5.plot(year,constants[4]*CO2,c='m',label='CO2/Anthropogenic Impact')  # Plots CO2 Surface Radiative Forcing

# Plot titles and labels
ax1.set_title('Regressed Global Temperature Anomaly($^\circ$C) vs Year')
ax2.set_title('Predictor Variables X Regression Coefficients')
ax1.set_ylabel('Temperature Anomaly\n($^\circ$C)')
ax1.legend(loc=0)
ax2.set_ylabel('Anomaly\n($^\circ$C)')
ax2.legend(loc=0)
ax3.set_ylabel('Anomaly\n($^\circ$C)')
ax3.legend(loc=0)
ax4.set_ylabel('Anomaly\n($^\circ$C)')
ax4.legend(loc=0)
ax5.set_ylabel('Anomaly\n($^\circ$C)')
ax5.legend(loc=0)
plt.savefig('Plot1.png')
plt.show()

###########################################
### Second Plot
###########################################

# Extracts variables
year_Full = data_org[:,0]  # year
tsi_Full = data_org[:,2]  # total_solar_irradiance_anomaly (W/m^2)
enso_Exp = data_org[:,3]  # enso_super_el_nino_2025
enso_Con = data_org[:,4]  # enso_flat_after_present
vol_Exp = data_org[:,5]  # volcanic_aer_big_eruption_2015
vol_Con = data_org[:,6]  # volcanic_aer_flat_after_present
CO2_Full = data_org[:,7]  # surface_radiative_forcing_co2 (W/m^2)

# Calculates the model fit result for both the control and experiment
temp_fut_Con = constants[0] + constants[1]*tsi_Full + constants[2]*enso_Con + constants[3]*vol_Con + constants[4]*CO2_Full
temp_fut_Exp = constants[0] + constants[1]*tsi_Full + constants[2]*enso_Exp + constants[3]*vol_Exp + constants[4]*CO2_Full

# Plots both control and experiment lines with correct labels and settings
plt.plot(year_Full,temp_fut_Con,color='blue',label='Flat')
plt.plot(year_Full,temp_fut_Exp,ls='--',color='red',label='Eruption + Super El Nino')

# Limits the x-axis to only show years from 1980-2030
plt.xlim(1980,2030)

# General plotting parts
plt.title('Global Surface Temperature Anomalies Projection (1980-2030)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly($^\circ$C)')
plt.legend(loc=0)
plt.savefig('Plot2.png')
plt.show()