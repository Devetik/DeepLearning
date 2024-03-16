import numpy as np
import pandas as pd

df = pd.read_csv("Universities.csv")
df.head()
    #                                 Sector                             University  Year  Completions Geography
    #0            Private for-profit, 2-year       Pima Medical Institute-Las Vegas  2016          591    Nevada   
    #1  Private for-profit, less-than 2-year       Healthcare Preparatory Institute  2016           28    Nevada   
    #2  Private for-profit, less-than 2-year              Milan Institute-Las Vegas  2016          408    Nevada   
    #3  Private for-profit, less-than 2-year  Utah College of Massage Therapy-Vegas  2016          240    Nevada   
    #4               Public, 4-year or above                 Western Nevada College  2016          960    Nevada

dfSum = df.groupby("Year").sum()
print(dfSum["Completions"])
                                                    #Year
                                                    #2012    20333
                                                    #2013    21046
                                                    #2014    24730
                                                    #2015    26279
                                                    #2016    26224
dfMean = df.groupby('Year')['Completions'].mean()
print(dfMean)
                                                    #Year
                                                    #2012    535.078947
                                                    #2013    526.150000
                                                    #2014    588.809524
                                                    #2015    597.250000
                                                    #2016    609.860465
                                                    
""" 
    count	    Nombre d'observations non null
    sum	        Somme des valeurs
    mean	    Moyenne des valeurs
    mad	        Écart moyen absolu
    median	    Médiane arithmétique des valeurs
    min	        Minimum
    max	        Maximum
    mode	    Mode
    abs	        Valeur absolue
    prod	    Produit des valeurs
    std	        Écart type non biaisé
    var	        Variance non biaisée
    sem	        Erreur standard de la moyenne non biaisée
    skew	    Asymétrie non biaisée (3e moment)
    kurt	    Kurtosis non biaisé (4ème moment)
    quantile	Quantile de l'échantillon (valeur en %)
    cumsum	    Somme cumulative
    cumprod	    Produit cumulé
    cummax	    Maximum cumulé
    cummin	    Minimum cumulé
"""

print(df.groupby(["Year", "Sector"]).sum())
#                                                                                     University  Completions                                          Geography
#Year Sector
#2012 Private for-profit, 2-year               Career College of Northern NevadaPaul Mitchell...         3072  NevadaNevadaNevadaNevadaNevadaNevadaNevadaNeva...
#     Private for-profit, 4-year or above      Sanford-Brown College-Las VegasITT Technical I...          632                           NevadaNevadaNevadaNevada
#     Private for-profit, less-than 2-year     Academy of Hair Design-Las VegasMilan Institut...         1327         NevadaNevadaNevadaNevadaNevadaNevadaNevada
#     Private not-for-profit, 2-year           Everest College-HendersonExpertise Cosmetology...          665                                       NevadaNevada
#     Private not-for-profit, 4-year or above  Roseman University of Health SciencesTouro Uni...         1059                                 NevadaNevadaNevada
#     Public, 2-year                                           Truckee Meadows Community College         1170                                             Nevada
#     Public, 4-year or above                  University of Nevada-RenoCollege of Southern N...        12408               NevadaNevadaNevadaNevadaNevadaNevada
#2013 Private for-profit, 2-year               Aviation Institute of Maintenance-Las VegasCar...         3053  NevadaNevadaNevadaNevadaNevadaNevadaNevadaNeva...
#     Private for-profit, 4-year or above      DeVry University-NevadaITT Technical Institute...          775                     NevadaNevadaNevadaNevadaNevada
#     Private for-profit, less-than 2-year     Academy of Hair Design-Las VegasMilan Institut...         1281         NevadaNevadaNevadaNevadaNevadaNevadaNevada
#     Private not-for-profit, 2-year           Expertise Cosmetology InstituteEverest College...          471                                       NevadaNevada
#     Private not-for-profit, 4-year or above  Roseman University of Health SciencesTouro Uni...         1016                                 NevadaNevadaNevada
#     Public, 2-year                                           Truckee Meadows Community College         1633                                             Nevada
#     Public, 4-year or above                  University of Nevada-RenoUniversity of Nevada-...        12817               NevadaNevadaNevadaNevadaNevadaNevada

print(df.groupby("Year").describe())
                                        #     Completions
                                        #           count        mean          std   min     25%    50%     75%     max
                                        #Year
                                        #2012        38.0  535.078947  1036.433239  13.0  114.25  229.5  420.50  5388.0
                                        #2013        40.0  526.150000  1040.474782   0.0   98.50  189.0  413.00  5278.0
                                        #2014        42.0  588.809524  1150.355857   0.0  104.50  203.5  371.75  5093.0
                                        #2015        44.0  597.250000  1183.371791   0.0   87.75  191.0  405.75  5335.0
                                        #2016        43.0  609.860465  1235.952796   0.0   90.00  208.0  414.00  5367.0
                                        
print(df.groupby("Year").describe().transpose())
                                        #Completions count    38.000000    40.000000    42.000000    44.000000    43.000000
                                        #            mean    535.078947   526.150000   588.809524   597.250000   609.860465
                                        #            std    1036.433239  1040.474782  1150.355857  1183.371791  1235.952796
                                        #            min      13.000000     0.000000     0.000000     0.000000     0.000000
                                        #            25%     114.250000    98.500000   104.500000    87.750000    90.000000
                                        #            50%     229.500000   189.000000   203.500000   191.000000   208.000000
                                        #            75%     420.500000   413.000000   371.750000   405.750000   414.000000
                                        #            max    5388.000000  5278.000000  5093.000000  5335.000000  5367.000000