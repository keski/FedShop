import pandas as pd
import os
from pathlib import Path
import glob
import time
import requests
import subprocess
import re

import sys
smk_directory = os.path.abspath(workflow.basedir)
print(smk_directory)
sys.path.append(os.path.join(Path(smk_directory).parent, "fedshop"))

from utils import ping, LOGGER, load_config

#===============================
# GENERATION PHASE:
# - Generate data
# - Ingest the data in virtuoso
# - Generate query instances
# - Generate expected results
# - Generate expected source selection
# - Generate expected metrics
#===============================

CONFIGFILE = config["configfile"]

WORK_DIR = "experiments/bsbm"
CONFIG = load_config(CONFIGFILE)
CONFIG_GEN = CONFIG["generation"]
CONFIG_EVAL = CONFIG["evaluation"]

SPARQL_COMPOSE_FILE = CONFIG_GEN["virtuoso"]["compose_file"]
SPARQL_SERVICE_NAME = CONFIG_GEN["virtuoso"]["service_name"]
VIRTUOSO_COMPOSE_CONFIG = load_config(SPARQL_COMPOSE_FILE)
SPARQL_CONTAINER_NAME = VIRTUOSO_COMPOSE_CONFIG["services"][SPARQL_SERVICE_NAME]["container_name"]
DOCKER_EXEC_CMD = f"docker exec {SPARQL_CONTAINER_NAME}"

PROXY_COMPOSE_FILE =  CONFIG_EVAL["proxy"]["compose_file"]
PROXY_SERVICE_NAME = CONFIG_EVAL["proxy"]["service_name"]
PROXY_CONTAINER_NAMES = CONFIG_EVAL["proxy"]["container_name"]
PROXY_SERVER = CONFIG_EVAL["proxy"]["endpoint"]
PROXY_PORT = CONFIG_EVAL["proxy"]["port"]
PROXY_SPARQL_ENDPOINT = PROXY_SERVER + "sparql"

CONTAINER_PATH_TO_ISQL = "/opt/virtuoso-opensource/bin/isql"
CONTAINER_PATH_TO_DATA = "/usr/share/proj/" 

# CONTAINER_PATH_TO_ISQL = "/usr/local/virtuoso-opensource/bin/isql-v" 
# CONTAINER_PATH_TO_DATA = "/usr/local/virtuoso-opensource/share/virtuoso/vad"

N_QUERY_INSTANCES = CONFIG_GEN["n_query_instances"]
VERBOSE = CONFIG_GEN["verbose"]
N_BATCH = CONFIG_GEN["n_batch"]

QUERY_DIR = f"{WORK_DIR}/queries"
MODEL_DIR = f"{WORK_DIR}/model"
BENCH_DIR = f"{WORK_DIR}/benchmark/generation"
TEMPLATE_DIR = f"{MODEL_DIR}/watdiv"


#=================
# USEFUL FUNCTIONS
#=================


#=================
# PIPELINE
#=================

rule all:
    input: 
        expand(
                "{benchDir}/{query}/instance_{instance_id}/batch_{batch_id}/rsa.sparql",
                benchDir=BENCH_DIR,
                query=[Path(os.path.join(QUERY_DIR, f)).resolve().stem for f in os.listdir(QUERY_DIR) if f.endswith(".sparql")],
                instance_id=range(N_QUERY_INSTANCES),
                batch_id=range(N_BATCH)
        )

rule generate_rsa_quries:
    priority: 7
    threads: 1
    input: 
        workload_instance="{benchDir}/{query}/instance_{instance_id}/injected.sparql",
        loaded_virtuoso=f"{WORK_DIR}/virtuoso-federation-endpoints-ok.txt",
    output: "{benchDir}/{query}/instance_{instance_id}/batch_{batch_id}/rsa.sparql"
    run: 
        shell("python fedshop/engines/rsa.py create-service-query {CONFIGFILE} {input.workload_instance} {output} --batch-id {wildcards.batch_id}")

rule instanciate_workload:
    priority: 7
    threads: 1
    input: 
        queryfile=expand("{queryDir}/{{query}}.sparql", queryDir=QUERY_DIR),
        workload_value_selection="{benchDir}/{query}/workload_value_selection.csv"
    output:
        injected_query="{benchDir}/{query}/instance_{instance_id}/injected.sparql",
        injection_cache="{benchDir}/{query}/instance_{instance_id}/injection_cache.json",
        prefix_cache="{benchDir}/{query}/instance_{instance_id}/prefix_cache.json"
    params:
        batch_id = 0
    run:
        shell("python fedshop/query.py instanciate-workload {CONFIGFILE} {input.queryfile} {input.workload_value_selection} {output.injected_query} {wildcards.instance_id}")
        
        in_injected_opt_query = f"{QUERY_DIR}/{wildcards.query}.injected.opt"
        out_injected_opt_query = f"{output.injected_query}.opt"

        if os.path.exists(in_injected_opt_query):
            shell(f"python fedshop/query.py inject-from-cache {in_injected_opt_query} {output.injection_cache} {out_injected_opt_query}")

rule create_workload_value_selection:
    priority: 8
    threads: 5
    input: 
        value_selection_query="{benchDir}/{query}/value_selection.sparql",
        value_selection="{benchDir}/{query}/value_selection.csv"
    output: "{benchDir}/{query}/workload_value_selection.csv"
    params:
        n_query_instances = N_QUERY_INSTANCES
    shell:
        "python fedshop/query.py create-workload-value-selection {input.value_selection_query} {input.value_selection} {output} {params.n_query_instances}"

rule exec_value_selection_query:
    priority: 9
    threads: 1
    retries: 2
    input: 
        value_selection_query="{benchDir}/{query}/value_selection.sparql",
        virtuoso_status="{benchDir}/virtuoso-ok.txt"
    output: "{benchDir}/{query}/value_selection.csv"
    run:
        endpoint_batch0 = CONFIG_GEN["virtuoso"]["batch_endpoints"][0]
        shell("python fedshop/query.py execute-query {input.value_selection_query} {output} {endpoint_batch0}")

rule build_value_selection_query:
    priority: 10
    threads: 5
    input: 
        queryfile=expand("{queryDir}/{{query}}.sparql", queryDir=QUERY_DIR)
    output: "{benchDir}/{query}/value_selection.sparql"
    shell: "python fedshop/query.py build-value-selection-query {input.queryfile} {output}"
