# AIWUM 2.1
Machine Learning codes for the USGS MAP project.

Authors: [Sayantan Majumdar](https://scholar.google.com/citations?user=iYlO-VcAAAAJ&hl=en) [sayantan.majumdar@dri.edu], [Ryan Smith](https://scholar.google.com/citations?user=nzSrr8oAAAAJ&hl=en) [ryan.g.smith@colostate.edu], and [Vincent E. White](https://www.usgs.gov/staff-profiles/vincent-e-white) [vwhite@usgs.gov]

<img src="Readme_Figures/USGS_logo.png" height="40"/> &nbsp; <img src="Readme_Figures/CSU-Signature-357-617.png" height="50"/> <img src="Readme_Figures/official-dri-logotag-trans-bkgd.png" height="45"/>

## Citations
**Software**: Majumdar, S., Smith, R., and White, V.E., 2023, Aquaculture and Irrigation Water Use Model 2.0 Repository: U.S. Geological Survey data release, https://doi.org…

**Data Release**: Majumdar, S., Smith, R.G., Hasan, M.F., Wilson, J.L., Bristow, E.L., Rigby, J.R., Kress, W.H., Painter, J.A., and White, V.E., 2023, Aquaculture and Irrigation Water Use Model (AIWUM) 2.0 input and output datasets, https://doi.org/10.5066/P9CET25K.

## Summary

The motivation for this project was to improve estimates of groundwater usage across the Mississippi Embayment (MISE), a large area within the Mississippi Alluvial Plain (MAP) region in support of an ongoing USGS effort to model the groundwater resources of the region. Agricultural use is the dominant water use in this region, and very few wells are monitored. The Mississippi Delta region has the most monitoring wells, with flowmeters on roughly 10% of the total irrigation wells. [Wilson (2021)](https://doi.org/10.3133/sir20215011) developed a lookup table based on these data that estimates water use based on average water use for each crop type, for specific regions, and precipitation amounts. The latest iteration of the [Wilson (2021)](https://doi.org/10.3133/sir20215011) model is referred to as AIWUM 1.1. The method developed here is referred to as AIWUM 2.0. 

The goal of this project was to improve on that method by using additional data that is likely related to water use, such as remotely-sensed evapotranspiration, model-based estimates of soil moisture, and temperature. These and other variables are likely related to water use, but quantifying this relationship, which is often complex and non-linear, with traditional models is a challenge. Machine learning provides robust tools for ingesting large numbers of predictor variables and quantifying how they are related to a prediction of interest ([Hastie et al., 2001](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf)). Previous works in Kansas and Arizona have demonstrated the ability to relate remote sensing and other gridded predictors to groundwater pumping data with good accuracy  (Majumdar et al., [2020](https://doi.org/10.1029/2020WR028059), [2022](https://doi.org/10.1002/hyp.14757)). In this study, we implement a similar approach to estimate groundwater pumping throughout the MAP, at an annual time-step, using data from existing flowmeters in Mississippi.

In this study, we use Distributed Random Forests (DRF) to solve a multi-variate regression problem wherein our target is to predict the groundwater use across the MISE from 2014-2020.  


Here we used the [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.5/) ([Ke et al., 2017](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)) Python library to implement the AIWUM 2 model and compared its performance against other algorithms, e.g., Gradient Boosting Trees, Support Vector Machine, Extremely Randomized Trees, etc.


The predictor variables include latitude, longitude, crop type, precipitation, maximum temperature (the average daily maximum from April - September), total evapotranspiration estimated with SSEBop, surface run-off and soil moisture (TerraClimate), infiltration rate (from Hydrologic Soil Group), and SWB Irrigation. A table summary of the predictor variables is given below.

| Variable                             | Operator               | Time Period Sampled | Spatial Resolution (m) | Source                                                                                                | Additional Processing Notes                                                                             |
|--------------------------------------|------------------------|---------------------|------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Latitude                             | ---                    | 2014-2020           | ---                    | Field Polygon                                                                                         | Used yearly permitted boundaries and k-Dimensional Tree to get field centroids                          |
| Longitude                            | ---                    | 2014-2020           | ---                    | Field Polygon                                                                                         | Used yearly permitted boundaries and k-Dimensional Tree to get field centroids                          |
| Crop type                            | Mode (spatial)         | Entire year         | 30                     | [Boryan et al. (2011)](https://doi.org/10.1080/10106049.2011.562309)                                  | Data were up-sampled to 0.1 km grid as in [Bristow and Wilson (2023)](https://doi.org/10.5066/P9RGZOBZ). |
| Precipitation                        | Sum (temporal)         | Apr - Sep           | 800                    | [Daly et al. (1997)](https://prism.oregonstate.edu/documents/pubs/1997appclim_PRISMapproach_daly.pdf) |                                                                                                         |
| Temperature                          | Median/Mean (temporal) | Apr - Sep           | 800                    | [Daly et al. (1997)](https://prism.oregonstate.edu/documents/pubs/1997appclim_PRISMapproach_daly.pdf) | Median of monthly average daily maximum temperature                                                     |
| Evapotranspiration                   | Sum (temporal)         | Apr - Sep           | 500                    | [Senay et al. (2013)](https://doi.org/10.1111/jawr.12057)                                             |                                                                                                         |
| Surface run-off                      | Mean (temporal)        | Apr - Sep           | ~ 4,000                | [Abatzoglou et al. (2018)](https://doi.org/10.1038/sdata.2017.191)                                    |                                                                                                         |
| Soil Moisture                        | Difference (temporal)  | Apr - Sep           | ~ 4000                 | [Abatzoglou et al. (2018)](https://doi.org/10.1038/sdata.2017.191)                                    | Difference in soil moisture between Apr and Sep                                                         |
| Infiltration Rate (derived from HSG) | ---                    | ---                 | 1000                   | [Westenbroeck et al. (2021)](https://doi.org/10.3133/ofr20211008)                                     | Derived from HSG following the average infiltration rate                                                |
| SWB Irrigation                       | Mean (temporal)        | Apr - Sep           | 1000                   | [Westenbroeck et al. (2021)](https://doi.org/10.3133/ofr20211008)                                                                        |                                                                                                         |

The figure below shows the general processing workflow.

![preview](Readme_Figures/Workflow_MAP.png)

## Getting Started

[Installing the correct environment and running the project](aiwum2/README.md)

## Related External Resources
Abatzoglou, J. T., Dobrowski, S. Z., Parks, S. A., & Hegewisch, K. C. (2018). TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958–2015. Scientific Data, 5(1), 170191. https://doi.org/10.1038/sdata.2017.191

Anaconda Software Distribution. (2023). Anaconda Documentation. Anaconda Inc. Retrieved from https://docs.anaconda.com/

Boryan, C., Yang, Z., Mueller, R., & Craig, M. (2011). Monitoring US agriculture: the US Department of Agriculture, National Agricultural Statistics Service, Cropland Data Layer Program. Geocarto International, 26(5), 341–358. https://doi.org/10.1080/10106049.2011.562309

Daly, C., & Taylor, G. (1997). The PRISM approach to mapping precipitation and temperature. 10th AMS Conf. on Applied Climatology, 20–23. https://prism.oregonstate.edu/documents/pubs/1997appclim_PRISMapproach_daly.pdf

Hastie, T., Tibshirani, R., & Friedman, J. (2001). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer New York.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In I. Guyon, U. V Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 30). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

Majumdar, S., Smith, R., Butler, J. J., & Lakshmi, V. (2020). Groundwater withdrawal prediction using integrated multitemporal remote sensing data sets and machine learning. Water Resources Research, 56(11), e2020WR028059. https://doi.org/10.1029/2020WR028059

Majumdar, S., Smith, R., Conway, B. D., & Lakshmi, V. (2022). Advancing remote sensing and machine learning‐driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. Hydrological Processes, 36(11), e14757. https://doi.org/10.1002/hyp.14757

Senay, G. B., Bohms, S., Singh, R. K., Gowda, P. H., Velpuri, N. M., Alemu, H., & Verdin, J. P. (2013). Operational Evapotranspiration Mapping Using Remote Sensing and Weather Datasets: A New Parameterization for the SSEB Approach. JAWRA Journal of the American Water Resources Association, 49(3), 577–591. https://doi.org/10.1111/jawr.12057

Westenbroek, S. M., Nielsen, M. ., & Ladd, D. E. (2021). Initial estimates of net infiltration and irrigation from a soil-water-balance model of the Mississippi Embayment Regional Aquifer Study Area: U.S. Geological Survey Open-File Report 2021-1008 (p. 29). US Geological Survey. https://doi.org/10.3133/ofr20211008

Wilson, J. L. (2021). Aquaculture and Irrigation Water-Use Model (AIWUM) Version 1.0—An Agricultural Water-Use Model Developed for the Mississippi Alluvial Plain, 1999–2017 (Scientific Investigations Report 2021-5011). U.S. Geological Survey. https://doi.org/10.3133/sir20215011