# Cluster job scripts (SGE/UGE examples)

These are the job scripts used for the paper's runs on a university GPU cluster
(Univa Grid Engine). They are kept as worked examples, not as portable launchers.
Before use, export the variables they reference and edit the `#$` directives
(queue name, log paths) for your scheduler:

```bash
export SKILLRC_ROOT=/path/to/SkillRC          # repository root
export SKILLRC_PY=/path/to/env/bin/python     # python with requirements.txt installed
export VLLM_ENV=/path/to/vllm-env             # env providing `vllm serve` (Qwen3-8B runs)
export HF_HOME=/path/to/hf_cache              # Hugging Face cache with Qwen/Qwen3-8B
export WEBSHOP_ENV=/path/to/webshop-env WEBSHOP_PY=$WEBSHOP_ENV/bin/python
```

Qwen runs start `vllm serve Qwen/Qwen3-8B` inside the job, wait for `/health`, and
point the harness at `http://localhost:<port>/v1` with
`chat_template_kwargs: {enable_thinking: false}`.
