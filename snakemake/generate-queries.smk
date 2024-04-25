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

from utils import ping, fedshop_logger, load_config
LOGGER = fedshop_logger(Path(__file__).name)

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

# Override settings if specified in the config file
BATCH_ID = str(config["batch"]).split(",") if config.get("batch") is not None else range(N_BATCH)
QUERY_PATH = (
    [Path(os.path.join(QUERY_DIR, f)).resolve().stem for f in str(config["query"]).split(",")] 
    if config.get("query") is not None else 
    [Path(os.path.join(QUERY_DIR, f)).resolve().stem for f in os.listdir(QUERY_DIR) if f.endswith(".sparql")]
)
INSTANCE_ID = str(config["instance"]).split(",") if config.get("instance") is not None else range(N_QUERY_INSTANCES)

DEBUG = eval(str(config["debug"])) if config.get("explain") is not None else False


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
                query=QUERY_PATH,
                instance_id=INSTANCE_ID,
                batch_id=BATCH_ID
        )

rule generate_rsa_queries:
    input: "{benchDir}/{query}/instance_{instance_id}/composition.json"
    output: "{benchDir}/{query}/instance_{instance_id}/batch_{batch_id}/rsa.sparql"
    params:
        workload_instance="{benchDir}/{query}/instance_{instance_id}/injected.sparql"
    run:
        shell("python fedshop/engines/rsa.py create-service-query {CONFIGFILE} {params.workload_instance} {output} --batch-id {wildcards.batch_id}")

rule decompose_query:
    threads: 1
    input: 
        workload_instance="{benchDir}/{query}/instance_{instance_id}/injected.sparql",
        loaded_virtuoso=f"{WORK_DIR}/virtuoso-federation-endpoints-ok.txt",
    output: "{benchDir}/{query}/instance_{instance_id}/composition.json"
    run: 
        shell("python fedshop/query.py decompose-query {input.workload_instance} {output}")

rule instanciate_workload:
    threads: 1
    input: 
        queryfile=expand("{queryDir}/{{query}}.sparql", queryDir=QUERY_DIR),
        workload_value_selection="{benchDir}/{query}/workload_value_selection.csv"
    output:
        injected_query="{benchDir}/{query}/instance_{instance_id}/injected.sparql",
        # injection_cache="{benchDir}/{query}/instance_{instance_id}/injection_cache.json",
        # prefix_cache="{benchDir}/{query}/instance_{instance_id}/prefix_cache.json"
    params:
        batch_id = 0
    run:
        shell("python fedshop/query.py instanciate-workload {input.queryfile} {input.workload_value_selection} {output.injected_query} {wildcards.instance_id}")
        
        # in_injected_opt_query = f"{QUERY_DIR}/{wildcards.query}.injected.opt"
        # out_injected_opt_query = f"{output.injected_query}.opt"

        # if os.path.exists(in_injected_opt_query):
        #     shell(f"python fedshop/query.py inject-from-cache {in_injected_opt_query} {output.injection_cache} {out_injected_opt_query}")

rule create_workload_value_selection:
    threads: 5
    input: 
        value_selection_infos="{benchDir}/{query}/value_selection.json"
    output: "{benchDir}/{query}/workload_value_selection.csv"
    params:
        n_query_instances = N_QUERY_INSTANCES,
    run:
        constfile = f"{QUERY_DIR}/{wildcards.query}.const.json"
        shell(f"python fedshop/query.py create-workload-value-selection {CONFIGFILE} {constfile} {input.value_selection_infos} {output} {params.n_query_instances}")

rule build_value_selection_query:
    threads: 5
    input: 
        constfile = expand("{queryDir}/{{query}}.const.json", queryDir=QUERY_DIR),
        queryfile = expand("{queryDir}/{{query}}.sparql", queryDir=QUERY_DIR)
    output: "{benchDir}/{query}/value_selection.json"
    shell: "python fedshop/query.py build-value-selection-query {input.queryfile} {input.constfile} {output}"
