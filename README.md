# Terra Research
Codebase for Terra Research. Projects span all layers of the Terra stack: blockchain, stability, dApps, e-commerce and beyond.
## Requirements
* Python 3.6 or above
* Basic python packages found in requirements.txt (numpy, pandas etc)

## Setup
The easiest and most reliable way to run the code is by setting up a python virtual environment. If you've never done this before, follow the straightforward instructions [here](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv). In addition, [virtualenvwrapper](https://docs.python-guide.org/dev/virtualenvs/#virtualenvwrapper) provides handy shortcuts for interracting with virtualenvs and helps organize them. Make sure you install Python 3.6 or above in your virtualenv.

After you activate your virtualenv, clone the research repo and run the following to install required dependencies:
```
pip install -r requirements.txt
```
You're all set!

## Projects
The codebase is organized around projects which are mostly independent code-wise. Projects are defined at the implementation level: two projects may be implementing different solutions to the same research problem, eg the mining rewards problem. Each project has its own directory. Project-wide code will be demarcated into separate directories.

### Project Directory

| Project       | Description   |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |
```
python simulation.py
```

#### estimate cost of equity for Luna
```
python cost_of_equity.py
```
