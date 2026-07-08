# LLM Prior BO



**Reference:**  "Large Language Models as Proxies for Human Experts in Synthesis Optimization"

<img src="./figs/fig1.png" alt="overview" width="500" height="250">

## Installation

Create anaconda environment for YAML configuration file

```
conda env create -f environment.yml
```
and enable it
```
conda activate LLMpiBO
```

## Running Scripts
Process the data
```
python 0_process_data.py
```
Process the papers
```
python 1_process_papers.py
```
Generate the priors using LLM calls
```
python 2_prior_generator.py
```
Run the LLM PiBO scheme
```
python 3_run_PiBO.py
```
