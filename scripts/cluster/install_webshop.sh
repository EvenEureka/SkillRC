#!/usr/bin/env bash
# Install WebShop into its OWN conda env (py3.8.13) — kept separate from skillrc to
# avoid the pydantic v1 (spacy 3.3) vs v2 (openai) clash. Harness talks to it over
# HTTP (scripts/webshop_server.py). Run in background on the login node (I/O-bound;
# does not need a GPU, so it won't compete with the retrieval-ablation jobs).
# Steps are individually non-fatal so one flaky gdown doesn't abort everything.
set -uo pipefail
CONDA=${CONDA_ROOT}
ENV=webshop
WROOT=${SKILLRC_ROOT}/third_party/WebShop
WPY=$CONDA/envs/$ENV/bin/python
WPIP=$CONDA/envs/$ENV/bin/pip
DATA="${DATA:-small}"

echo "=== [1/6] create env py3.8.13 ==="
$CONDA/bin/conda create -y -n $ENV python=3.8.13 2>&1 | tail -3

echo "=== [2/6] pip requirements ==="
cd "$WROOT"
$WPIP install -r requirements.txt 2>&1 | tail -12 || echo "WARN pip reqs"

echo "=== [3/6] conda faiss-cpu + openjdk 11 ==="
$CONDA/bin/conda install -y -n $ENV -c pytorch faiss-cpu 2>&1 | tail -3 || echo "WARN faiss"
$CONDA/bin/conda install -y -n $ENV -c conda-forge openjdk=11 2>&1 | tail -3 || echo "WARN jdk"

echo "=== [4/6] spaCy en_core_web_lg ==="
$WPY -m spacy download en_core_web_lg 2>&1 | tail -3 || echo "WARN spacy model"

echo "=== [5/6] download data ($DATA) ==="
mkdir -p "$WROOT/data"; cd "$WROOT/data"
GDOWN=$CONDA/envs/$ENV/bin/gdown
if [ "$DATA" = "small" ]; then
  $GDOWN https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib 2>&1 | tail -2 || echo "WARN dl1"
  $GDOWN https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu 2>&1 | tail -2 || echo "WARN dl2"
else
  $GDOWN https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB 2>&1 | tail -2 || echo "WARN dl1"
  $GDOWN https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi 2>&1 | tail -2 || echo "WARN dl2"
fi
$GDOWN https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O 2>&1 | tail -2 || echo "WARN dl_humanins"

echo "=== [6/6] build Lucene index ==="
cd "$WROOT/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
$WPY convert_product_file_format.py 2>&1 | tail -5 || echo "WARN convert"
# env's java (openjdk11) + python must lead PATH for pyserini
PATH=$CONDA/envs/$ENV/bin:$PATH bash run_indexing.sh 2>&1 | tail -15 || echo "WARN indexing"
cd "$WROOT"
echo "=== verify import ==="
$WPY -c "import web_agent_site, pyserini, faiss, spacy; print('webshop imports OK')" 2>&1 | tail -3 || echo "WARN import"
echo "WEBSHOP_INSTALL_DONE"
