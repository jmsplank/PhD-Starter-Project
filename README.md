## Install Requirements

Code written in Python 3.8.2.

`python3 -m venv env`  
`source env/bin/activate`  
`pip install -r requirements.txt`

### Modifications to pyspedaS

The `local_data_dir` for mms was changed to point to mms_data folder inside the project:

```
Change: PhD-Starter-Project/env/lib/python3.8/site-packages/pyspedas/mms/mms_config.py
Line 3:
CONFIG = {'local_data_dir': '<PATH_TO_REPO>/PhD-Starter-Project/mms_data',
```

## Running the code

`python reproduce_fig_2.py`
