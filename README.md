# Terra Research

# Requirements
* Python 3.6 or above
* Basic python packages found in requirements.txt (numpy, pandas etc)

# Setup
The easiest and most reliable way to run the code is by setting up a python virtual environment. If you've never done this before, follow the straightforward instructions [here](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv). In addition, [virtualenvwrapper](https://docs.python-guide.org/dev/virtualenvs/#virtualenvwrapper) provides handy shortcuts for interracting with virtualenvs and helps organize them. Make sure you install Python 3.6 or above in your virtualenv.

After you activate your virtualenv, clone the research repo and run the following to install required dependencies:
```
pip install -r requirements.txt
```
You're all set!

# run simulation
```
python simulation.py
```

# estimate cost of equity for Luna
```
python cost_of_equity.py
```
