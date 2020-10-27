## Install Requirements

Code written in Python 3.8.2.

1. Create a virtual environment:
```bash
pip3 install virtualenvwrapper
source virtualenvwrapper.sh
mkvirtualenv PhD-Starter-Project-env
workon PhD-Starter-Project-env
```

2. Clone repository:
```bash
git clone https://github.com/jmsplank/PhD-Starter-Project
cd PhD-Starter-Project
```

3. Install Requirements (Note: pybowshock will not install without permission)
```bash
pip install -r requirements.txt
```

4. (Optional) Modify pyspedas download directory
```bash
cd ~/.virtualenvs/PhD-Starter-Porject-env/lib/python3.8/site-packages/pyspedas/mms/
vi mms_config.py
```
And modify the `local_data_dir` parameter (Note: Use full path as ~ <HOME> etc. may not work)

## Running the code

Most scripts are self contained. Use `python <PATH_TO_SCRIPT>` to run the script. E.g. `python shock_normal/timing_analysis.py`
