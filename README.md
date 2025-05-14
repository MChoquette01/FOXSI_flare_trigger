# FOXSI_flare_trigger
Fun data science work for FOXSI-5 (Focusing Optics X-ray Solar Imager)!

### Problem Statement

Solar flares are very large eruptions on the surface of the Sun which release radiation into space. FOXSI-5 is a sounding rocket payload which seeks to observe a solar flare as part of a dedicated solar flare campaign. However, while large, solar flares are transient events and can approach with little warning. This project leverages NOAA X-ray data to train and analyze machine learning models to predict maximum X-ray flux 8-14 minutes out (when FOXSI-5 would be ready to make an observation) while minimizing launches which do not observe the desired level of flux.

This work focused on using decision trees and their cousins (random forests and gradient-boosted trees.) There's still much unknown about how solar flares develop, so it's desirable to analyze a successful model for why it makes the decisions and predictions that it does - a quest made easier with decision trees. 

### Prerequisites

This repo uses a FITS file of solar flares create by making_historical.py in the parameter_search repo (https://github.com/pet00184/parameter_search). These flare events are parsed into a local MongoDB database.

#### MongoDB
The Community edition can be downloaded here: https://www.mongodb.com/try/download/community.

##### MSI
Running these scripts that read/write to the MongoDB on the Minnesota Supercomputer Institute is desirable. Unfortunately, permissions prevent rank and file users from installing the executable directly at MSI. Mongo dies provide a Docker container that can be run locally. At MSI, run

apptainer pull docker://mongo:latest

Create a local folder to store the DB, for example, "FlaresDB"

mkdir FlaresDB

Then start the Mongo instance

apptainer instance start --bind $PWD/FlaresDB:/data/db mongo_latest.sif mongo_1
apptainer run instance://mongo_1 2>&1 > /dev/null &

The end of the second command suppress output so the terminal input is still accessible while Mongo is running. At this point, run any scripts/commands to interact with MongoDB. The instance can be stopped by running

apptainer instance stop mongo_1

### How to run

First, data should be downloaded from the NOAA: https://www.ncei.noaa.gov/products/goes-r-extreme-ultraviolet-xray-irradiance
Download XRS 1-minute averages: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/
And a flare sumamry file: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-flsum_science/
Both should be from the same satellite and the same date range!

X-ray flux data in windows ranging from 15 before the start of a flare to 15 after the end using the flare summary file can be extracted using the parameter_search repo. making_historical.py will create a GOES_XRS_historical.fits file containing information for each flare.

#### FITS to Mongo

Run Utilities/WriteFalreFITSToDb.py, using the above FITS file and 1-minute averages as input. This will create a MongoDB database with one record corresponding to data from one flare at one point in time, time relative to the start of the flare. utilities/MSIMongoTest.py can be used to verify records exist in the databse as expected and be used as a sandbox for simple queries. Note there are also some scripts to write the same data to a SQL database - these were written during development but not often used. Scripts elsewhere in the pipeline currently pull data from MongoDB only.

#### Linear Interpolation

The raw data contains some missing values. Most of these are in temperature and emission measure n-minute differences, however a few usually exist in the raw X-ray flux measurements as well. These values are resolved using linear interpolation. Running LinearInterpolator.py creates look up tables with linearly interpolated data for each flare. (During model training, each model should only have access to data from that point in time, so lookup tables allow for interpolated values to be plugged in on the fly without looking forwards or backwards in time during model training.)

#### Model Training

Model training is done using GridSearch.py.

#### Model Pruning

To avoid potential overfitting, trained models went through a Cost-Complexity Pruning process. This involves a hyperparmeter which balances the complexity (depth) of part of a tree versus how effective it is at distinguishing classes. Several different hyperparmeters are tested, and the one which improves both precision and recall while improving precision the most is saved as the best pruned model. If no such model exist for all hyperparameters, the orignal model is saved off. This is done by CostComplexityPruning.py.

#### Applying Cancellation Policy

In practice, 'real-time' X-ray data is served on a 3-minute latency. If a launch is opted for, there is a 3-minute countdown followed by 2-minutes of travel time and a 6-minute observation window. Only one launch is permitted, so should a launch be opted for but further data suggest the flare will not be as strong as anticipated, the launch can be cancelled to preserve the window. Programmatically, this was accounted for by comparing the XRSA flux at the time of prediction to the XRSA flux 3 minutes out (at teh end of teh countdown.) If it is less at that point, the model prediction is overruled to be <C5. This process is done by ApplyCancellationPolicy.py.

#### Model Evaluation/Graphing

Most model perforamnce can be gauged by the output of Plots/ImprovementComparison/py. This will graph the incremental improvements in gained in pruning and application of the cancellation policy, as well as precision, recall and F1 for models of different types/architectures.

#### A Note on Variables

Analysis uses XRSA and XRSB fluxes, temperature, emission measure and 1-5 minute differences of each. However, as the temperature and emission mesasure differences tend to have many NaN values (which are resolved through linear interpolation but still represent lost information), new variables were defined which calcualte the direct temperature and emission measure differences. In graphs and other saved output, the 'classic colar physicsist' method of caluclating temperature and emission measure differences based on X-ray flux is refered to as '...From XRS Difference', whereas the direct method is simply called '3-Minute Temperature Difference' or '5-Minute Emission Measure Difference'. Unfortunately, this naming conention was late in coming, so many parts of the code refer to the X-ray based measurements as '3-Minute Temperature Difference' and refer to the direct method as 'Naive 3-Minute Temperature Difference.' This is a bit jargony so was dropped before final analysis was done. 


### Scripts

ApplyCancellationPolicy.py: Applies a simulated launch cancellation policy to results
CompareScores.py: Creates graphs comparing performance metrics for two different models
CostComplexityPruning.py: Prune trained models
GridSearch.py: The main model training script. Finds the optimal performing tree given a scoring metric and range of values for each hyperparameter.
LinearInterpolator.py: Creates lookup tables of temporally linearly interpolated data to fill missing values.
TemporalAnalysis.py: Conduct analysis on a second holdout set to gauge model performnce across all models in the ensemble.
tree_common.py: Common functions used to tree operations (creating trees, splitting into train/tests sets, stc.)



GridSearch.py: Finds the optimal performing tree given a scoring metric and range of values for each hyperparameter.
GraphTreeDiagnostics.py: Creates visualizations for the results of GridSearch.py
GraphNodeResults.py: Examines results at a more granular level than the above script, such as exploring decision confidence
tree_common.py: Common functions used to tree operations (creating trees, splitting into train/tests sets, stc.)

### What's 'ProbabalisticPredictor?'

At one time there was an idea to take live data (like what was used above) and find statistically similar light curves and report those out as likely predictions. However, choosing which parmaeters to focus on and how to define similar need to be more scoped out. Feature importance graphs create for the above models could prove a good starting point!