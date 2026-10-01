# macOS quick start

## 1. Check Python

Open Terminal and run:

```bash
python3 --version
```

Python 3.10 or later is required.

If `python3` is not found, install Python from python.org or with Homebrew, then reopen Terminal.

## 2. Open the project folder in Terminal

If the unzipped project is in Downloads:

```bash
cd ~/Downloads/portfolio-risk-lab
```

Tip: type `cd ` including the space, then drag the project folder from Finder into the Terminal window and press Return.

## 3. Create an isolated environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When the environment is active, Terminal normally shows `(.venv)` at the start of the line.

## 4. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The first installation requires internet access to download Python packages.

## 5. Run the baseline project

```bash
python run_demo.py
```

Then open the report:

```bash
open outputs/demo/report.html
```

## 6. Run the 250-market robustness experiment

```bash
python run_robustness.py --experiments 250
```

Open the report:

```bash
open outputs/robustness/robustness_report.html
```

For a quick test first:

```bash
python run_robustness.py --experiments 25
```

## 7. Run the sensitivity analysis

```bash
python run_sensitivity.py
```

The CSV files are saved under `outputs/sensitivity/`.

## 8. Run automated tests

```bash
python -m pytest -q
```

This release should report:

```text
22 passed
```

## 9. Leave the environment

```bash
deactivate
```

## If something fails

Confirm that you are inside the repository folder and that `(.venv)` is visible. Then run:

```bash
python --version
python -m pip --version
python -m pip install -e ".[dev]"
python -m pytest -q
```
