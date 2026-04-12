# RoboJuDo Minimal

This repository has been reduced to the minimum sim2sim path for the custom G1 policy:

- config: `g1_my_rl`
- entrypoint: `scripts/run_pipeline.py`
- policy: `robojudo/policy/my_custom_policy.py`
- model: `assets/models/g1/my_custom/`

## Environment

Use the existing conda environment:

```bash
source /Users/luoxinyuan/miniforge3/etc/profile.d/conda.sh
conda activate robojudo
```

## Run

```bash
python scripts/run_pipeline.py
```

Equivalent explicit form:

```bash
python scripts/run_pipeline.py -c g1_my_rl
```

## Smoke Test

```bash
python scripts/test_custom_policy.py -c g1_my_rl
```

This checks config loading, model loading, observation generation, and action inference for the custom policy.
