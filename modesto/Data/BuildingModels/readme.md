# Building model data

This folder contains a range of model parameters for different buildings and different building models.

## `buildParamSummary.csv`
This file contains model parameters as defined by Reynders (2015) for RC models with day and night zone.
They are used for example in van der Heijde _et al._ (2017).

To be used by `LTIModels.RCmodels.RCmodel`.

## Folder `TEASER`
These files are model parameters for buildings in selected neighbourhoods in the city of Genk, courtesy of Ina
De Jaeger. The model parameter files are organized per neighbourhood and per street (first and second level).
These parameters can be read by

The folder `Archetypes` contains a single file with three archetypes for every neighbourhood. These archetypes
are constructed as the average building for all buildings that follow this archetype. Since the construction types
are the same for every archetype (based on the construction year), the archetypes can be found by averaging the
building areas.

To be used by `LTIModels.RCmodels.TeaserFourElement`


## References
Reynders, G. (2015). Quantifying the impact of building design on the potential of structural strorage for active demand response in residential buildings, (September), 266. https://doi.org/10.13140/RG.2.1.3630.2805

van der Heijde, B., Sourbron, M., Vega Arance, F. J., Salenbien, R., & Helsen, L. (2017). Unlocking flexibility by exploiting the thermal capacity of concrete core activation. Energy Procedia, 135, 92–104. https://doi.org/10.1016/j.egypro.2017.09.490