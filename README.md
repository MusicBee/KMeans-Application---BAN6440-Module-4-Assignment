# Module 4 – K-Means Weather Analysis - Kayode Ogunyemi



## Project Overview
This project analyses historical weather data from weather stations in Texas, USA, using data from the NOAA National Centers for Environmental Information (NCEI) Global Historical Climatology Network Daily (GHCN-Daily).
The analysis covers the period **2020–2024** and focuses on three primary weather variables:

- TMAX – Daily maximum temperature
- TMIN – Daily minimum temperature
- PRCP – Daily precipitation
The project combines data acquisition, data-quality assessment, exploratory data analysis, statistical analysis, and K-Means clustering to identify groups of weather stations with similar climate characteristics.



## Objectives
The main objectives of the project are to:

1. Acquire historical weather-station data from the GHCN-Daily dataset.
2. Identify Texas weather stations containing TMAX, TMIN, and PRCP observations.
3. Assess station data completeness and quality.
4. Select a geographically representative sample of 50 qualified stations.
5. Perform exploratory data analysis.
6. Analyse temperature and precipitation trends from 2020 to 2024.
7. Identify extreme weather events.
8. Apply K-Means clustering to group stations according to their weather characteristics.
9. Evaluate the quality of the clustering using silhouette analysis.
10. Validate the implementation using automated unit tests.






## Project Structure - This Assignment has been structured below
Module_4_KMeans_Weather_Analysis/
│
│
├── docs/
│   ├── executive_summary.docx
│   └── ai_disclosure.docx
│
├── outputs/
│   ├── eda/
│   ├── statistical_analysis/
│   └── kmeans_clustering/
│
├── src/
│   ├── config.py
│   ├── data_acquisition.py
│   ├── data_quality.py
│   ├── weather_analysis.py
│   ├── statistical_analysis.py
│   └── kmeans_clustering.py
│
├── tests/
│   ├── __init__.py
│   └── test_clustering.py
│
├── README.md
├── requirements.txt







## Dataset
Source - NOAA National Centers for Environmental Information (NCEI)
Global Historical Climatology Network – Daily (GHCN-Daily)
The project uses daily observations from weather stations located in Texas.
### Study Period
Start: 2020-01-01
End:   2024-12-31




### Weather Elements
| Element | Description | Unit |
| TMAX | Daily maximum temperature | °C |
| TMIN | Daily minimum temperature | °C |
| PRCP | Daily precipitation | mm |
Comments - GHCN temperature values are converted from tenths of degrees Celsius to degrees Celsius, while precipitation values are converted to millimetres where required by the source format.




## Data Selection and Quality
The initial station discovery identified 949 Texas candidate stations containing the required weather elements.
An 80% completeness threshold was applied to identify stations suitable for analysis.

The quality assessment produced:
- 949 candidate stations
- 304 qualified stations
- 130 stations with low completeness
- 34 stations with partial data
- 481 stations with no usable data for the study period

Comments - A final sample of 50 stations was selected using geographic farthest-point sampling to improve spatial coverage across the qualified Texas station network.
The final sample had approximately **97% average completeness** across TMAX, TMIN, and PRCP.




## Exploratory Data Analysis
The exploratory analysis examines:

- Temperature distributions
- Precipitation distributions
- Annual temperature patterns
- Monthly temperature patterns
- Monthly precipitation patterns
- Station-level variation
- TMAX/TMIN relationships
- Extreme temperature events
- Extreme precipitation events

    ### Key Findings
    Between 2020 and 2024:
    - Average TMAX increased from approximately **26.23°C to 27.05°C**.
    - Average TMIN increased from approximately **12.39°C to 13.59°C**.
    - Total precipitation across the selected station network increased from approximately **30,805.5 mm to 37,563.2 mm**.
    - 2024 had the highest average TMAX.
    - 2024 had the highest total precipitation.
    - 2022 had the lowest average TMIN and the lowest total precipitation.



## Statistical Analysis
The statistical analysis includes:
- Year-over-year changes
- 2020–2024 overall changes
- Linear trend analysis
- Seasonal analysis
- Station variability
- Temperature correlations
- Extreme-event counts

The estimated linear trends were:
| Variable | Trend |
| TMAX | +0.287°C/year |
| TMIN | +0.318°C/year |
| PRCP | +1,117.15 mm/year |

Comments - These trends describe the selected station sample during the study period and should not be interpreted as definitive long-term climate trends for the entire state of Texas.




## K-Means Clustering
K-Means clustering was applied at the weather-station level rather than to individual daily observations.
Eight features were used:

1. `tmax_mean`
2. `tmin_mean`
3. `temperature_range`
4. `prcp_total`
5. `tmax_std`
6. `tmin_std`
7. `prcp_std`
8. `tmax_tmin_correlation`

The features were standardised using `StandardScaler` before clustering. K-Means was evaluated for values of K from 2 to 10
The selected solution was:
###Results
K = 3
Silhouette Score = 0.3618

The three clusters represent broad data-driven weather-station profiles:
### Cluster 0
Warmer, Dry, Moderate Temperature Range

### Cluster 1
Moderate, Dry, Large Temperature Range

### Cluster 2
Moderate, Wet, Small Temperature Range
The silhouette score indicates measurable but moderate separation between the clusters. Therefore, the clusters should be interpreted as analytical groupings rather than definitive climate regions.





## Principal Component Analysis
PCA was used to visualise the clustering results in two dimensions.
The first two principal components explained approximately:

##Results
PC1 = 55.67%
PC2 = 26.22%
Combined = 81.89%

Justification - PCA was used primarily for visualisation. The original standardised features were retained for the K-Means clustering process.




## Testing
The project includes automated unit tests for the clustering implementation.
The tests cover:

- Dataset availability
- Required clustering features
- Missing-value checks
- Feature standardisation
- K-Means label generation
- Expected cluster count
- Silhouette-score validity
- Cluster-label completeness




## Bias, Fairness and Limitations
Several limitations should be considered when interpreting the results.

Weather-station coverage is not uniform across Texas. Applying an 80% completeness threshold can exclude stations with less complete records, which may affect geographic representation.

The final 50 stations were selected using geographic sampling to improve spatial coverage, but this does not eliminate all possible selection bias.

The clustering model uses eight weather-derived features and does not include potentially important variables such as:

- Elevation
- Humidity
- Wind
- Land cover
- Solar radiation

K-Means is also a distance-based algorithm and does not directly model the physical meteorological processes that produce weather patterns.
Therefore, the resulting clusters should be viewed as **data-driven station profiles**, rather than official or definitive climate classifications.





## AI Usage
ChatGPT (OpenAI) was used as a supporting AI tool during the project for:

- Brainstorming
- Research support
- Python development
- Debugging
- Data-analysis guidance
- Statistical interpretation
- K-Means methodology
- Code documentation
- Critical review of methodological limitations
- Bias and fairness discussion

AI-generated suggestions were reviewed against the actual dataset, program output, unit-test results, and external authoritative sources before being incorporated into the project.




## External Verification
The project used the following authoritative source to verify information about the weather dataset:

NOAA National Centers for Environmental Information (NCEI)
Global Historical Climatology Network – Daily (GHCN-Daily)
This source was used to verify the dataset structure, station observations, and definitions of weather elements including TMAX, TMIN, and PRCP.



## Author
Kayode Ogunyemi 

Module 4 – Data Analytics / K-Means Weather Analysis
