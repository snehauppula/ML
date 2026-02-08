# ML Project

Machine Learning project repository.

Important points

1..gitignore 

->Tells Git which files/folders NOT to track
->Prevents pushing:
-Virtual environments
-Secrets (API keys)
-Cache & temporary files
->Keeps the repo clean and safe

2.requirements.txt → Lists libraries needed to run the project

3.setup.py →Used to package your project as a reusable Python module.

4.__init__.py → Marks a folder as a Python package[Makes a folder importable]

Run → requirements.txt | Share → setup.py | Organize → __init__.py 

5. "- e ." in requiremnets .txt(Link to setup.py-> tells us setup.py exists)
-e → editable mode
. → current project
-Installs the project so code changes apply instantly.

6.components -Modules used in our project
-init-py-Makes components folder as package -importable and exportable
-Data-ingestion:process of collecting and importing data from various sources into a system for storage and analysis.
-Data-Transformation:process of converting raw data into a clean and suitable format for analysis or machine learning.
-Model-trainer:all the training process

7.
pipeline:A sequence of connected steps where data flows automatically from one stage to the next.

8.-logger.py
👉 Keeps a record of what your program is doing
(started, running, error happened)
🧠 Like a diary

-utils.py
👉 Stores helper functions used again and again
🧠 Like a toolbox

-exception.py
👉 Shows clear, custom error messages
🧠 Like warning labels

9.SYS Library: A built-in Python library that lets your program talk to the system (OS + Python runtime).

10.
What is StandardScaler?
It normalizes numeric data using this formula:
z=(x−mean)/standarddeviation
Because many ML models (like Logistic Regression, SVM, KNN) work better when:
Mean = 0
Standard deviation = 1
Example:
If Salary values:
20000, 30000, 40000
After scaling:
-1, 0, +1
Now model won’t be biased toward large numbers.

11.
OneHotEncoder
It converts categorical text data into numbers.
Example:
City column:
Hyd
Mum
Hyd
After OneHotEncoding:
City_Hyd	City_Mum
1	            0
0	            1
1	            0
Machine learning models cannot understand text — so we convert it into 0s and 1s.

12. ColumnTransformer → applies correct process to correct columns automatically
| Age | Salary | City_Hyd | City_Mum |
| --- | ------ | -------- | -------- |
| -1  | -1     | 1        | 0        |
| 1   | 1      | 0        | 1        |
