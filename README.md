# Nested Policy Pipeline

A hierarchical, LLM-driven framework for iteratively generating and optimizing battery-charging policies in a simplified microgrid.  At each meta-step, a “Task Generator” (Deepseek-R1) reasons over past performance and hyperparameters to craft a prompt, then a “Code Generator” (Qwen2.5) synthesizes a new policy class.  The result: simulated cost savings of up to 15% vs. a battery-off baseline.

---

## Features

- **Meta-Reinforcement Loop**: Alternating “reason → code → filter” steps  
- **LLM Roles**: Deepseek-R1 for prompt planning, Qwen2.5 for code synthesis  
- **Demand-Driven Simulation**: Constant or custom demand series  
- **Efficiency & Constraints**: Charge/discharge rate limits, one-way efficiencies  
- **Grid Export Revenue**: Earn revenue when prices go negative  
- **Baseline Comparison**: “Battery off” run for % cost-saving metrics  
- **Automated Retries**: Exponential back-off on LLM timeouts, error-aware prompt refinement  
- **Modular Codebase**: Python 3.9+, numpy, requests, matplotlib, python-dotenv  
- **Test Suite**: pytest-backed for algorithms, filters, meta-controller, policies

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Whisker2257/nested_policy_pipeline.git
   cd nested_policy_pipeline
2. **Create & activate a virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
3. **Install dependencies
   ```bash
   pip install -r requirements.txt
4. **Populate .env with your API keys
5. **Tweak model / sampling settings or simulation horizon
6. **Run the baseline + nested-policy pipeline and plot cost-savings
   ```bash
   python -m src.main

Nashe Gumbo

