This contains the evaluation of WOSP-C'24 submission titled "Privacy-Preserving Sharing of Data Analytics Runtime Metrics for Performance Modeling".

The entire evaluation can be run by executing the script `privacy_preservation_evaluation.py`. It works with Python 3.11 or higher.

It generates the results data in csv format if the files don't already exist and always generates the plots seen in the paper based on the available results.

To make sure the required libraries are installed and in compatible versions, create a virtual environment and install the libraries listed in `requirements.txt` via pip.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python privacy_preservation_evaluation.py
```

The full execution, generating the results of all three evaluation experiments (with the default 10 iterations) lasts about 2 hours on a laptop.
