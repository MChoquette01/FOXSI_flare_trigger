# FOXSI_flare_trigger
Fun data science work for FOXSI-5!

### Prerequisites

This repo uses a FITS file of solar flares create by making_historical.py in the parameter_search repo. These flare events are parsed into a local MongoDB database.

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

### Scripts
GridSearch.py: Finds the optimal performing tree given a scoring metric and range of values for each hyperparameter.
GraphTreeDiagnostics.py: Creates visualizations for the results of GridSearch.py
GraphNodeResults.py: Examines results at a more granular level than the above script, such as exploring decision confidence
tree_common.py: Common functions used to tree operations (creating trees, splitting into train/tests sets, stc.)