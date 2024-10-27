## Clone project
```shell
git clone https://github.com/ktxdev
```
## Environment Setup
### Required
python 3.11 **or** conda 24.9
### Create & Activate Virtual Environment
```shell
cd /path/to/your/project
```
#### Option 1: Using Python's venv
##### Create the environment
```shell
python -m venv env
```
##### Activate the environment on Windows
```shell
.\env\Scripts\activate
```
##### Activate the environment on MacOS/Linux
```shell
source env/bin/activate
```
##### Deactivate the environment
```shell
deactivate
```
#### Option 2: Using Conda
##### Create the environment
```shell
conda create --name env_name
```
**NB:** Replace ```env_name``` with your preferred name
##### Activate the environment
```shell
conda activate env_name
```
##### Deactivate the environment
```shell
conda deactivate
```
### Install required libraries
```shell
pip install -r requirements.txt
```
- Experiment 1 and 2 in the experiments log file used MinMaxScaler
and experiment 3-4 used StandardScaler
- 