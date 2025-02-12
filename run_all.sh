PORT=8501
curl -X POST -d "prefix=${WORKSPACE_BASE_URL}/port/${PORT}" -d "strip_prefix=true" http://localhost:9001/${PORT}

conda activate evocell

streamlit run ./evocell/app/main.py

#https://your_oneai_workbench/port/8501/docs

#https://apps.workbench.p171649450587.aws-amer.sanofi.com/apm0074851-rnaseq-amer01/mb-scrna/port/8501/docs
