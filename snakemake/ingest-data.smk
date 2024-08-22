import json

import numpy as np
import os
from pathlib import Path
import re

import sys
smk_directory = os.path.abspath(workflow.basedir)
sys.path.append(os.path.join(Path(smk_directory).parent, "fedshop"))

from utils import ping, fedshop_logger, load_config
from itertools import product
from omegaconf import OmegaConf
import time
import requests

LOGGER = fedshop_logger(Path(__file__).name)

CONFIGFILE = config["configfile"]

WORK_DIR = "experiments/bsbm"
QUERY_DIR = f"{WORK_DIR}/queries"
MODEL_DIR = f"{WORK_DIR}/model"
DATA_DIR = f"{MODEL_DIR}/dataset"
BENCH_DIR = f"{WORK_DIR}/benchmark/generation"
TEMPLATE_DIR = f"{MODEL_DIR}/watdiv"

CONFIG = load_config(CONFIGFILE)
CONFIG_GEN = CONFIG["generation"]
CONFIG_EVAL = CONFIG["evaluation"]

USE_DOCKER = CONFIG["use_docker"]

SPARQL_COMPOSE_FILE = CONFIG_GEN["virtuoso"]["compose_file"]
VIRTUOSO_COMPOSE_CONFIG = load_config(SPARQL_COMPOSE_FILE)
VIRTUOSO_PROXY_MAPPING_FILE = CONFIG_GEN["virtuoso"]["proxy_mapping"]
VIRTUOSO_PROXY_MAPPING_FILE_PREFIX = Path(VIRTUOSO_PROXY_MAPPING_FILE).stem
VIRTUOSO_PROXY_MAPPING_FILE_EXT = ".json"

SPARQL_SERVICE_NAME = CONFIG_GEN["virtuoso"]["service_name"]
SPARQL_CONTAINER_NAME = f"docker-{SPARQL_SERVICE_NAME}"
SPARQL_DEFAULT_ENDPOINT = CONFIG_GEN["virtuoso"]["default_endpoint"]

DOCKER_EXEC_CMD = f"docker exec {SPARQL_CONTAINER_NAME}"

# Modify the path to the Virtuoso ISQL executable and the path to the data
VIRTUOSO_PATH_TO_ISQL = CONFIG_GEN["virtuoso"]["isql"]
VIRTUOSO_PATH_TO_DATA = CONFIG_GEN["virtuoso"]["data_dir"]

if USE_DOCKER:
    VIRTUOSO_PATH_TO_ISQL = "/opt/virtuoso-opensource/bin/isql"
    VIRTUOSO_PATH_TO_DATA = "/usr/share/proj/" 

N_QUERY_INSTANCES = CONFIG_GEN["n_query_instances"]
VERBOSE = CONFIG_GEN["verbose"]
N_BATCH = CONFIG_GEN["n_batch"]

global NET_PORT

# Uncomment this line if starting from scratch
NET_PORT = 34200

#=================
# USEFUL FUNCTIONS
#=================
def validate(filename):
    with open(filename, "w") as f:
        f.write("ok")

def ping(endpoint):
    try:
        req = requests.get(endpoint)
        return req.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_data_files(wildcards):
    n_vendor = 10 * (int(wildcards.batch_id) + 1)
    n_ratingsite = 10 * (int(wildcards.batch_id) + 1)
    
    return [ f"{DATA_DIR}/vendor{vendor_id}.nq" for vendor_id in range(n_vendor) ] + [ f"{DATA_DIR}/ratingsite{ratingsite_id}.nq" for ratingsite_id in range(n_ratingsite) ]

#=================
# PIPELINE
#=================

rule all:
    input: 
        expand(
            "{workDir}/virtuoso-federation-endpoints-batch{batch_id}-ok.txt",
            workDir=WORK_DIR,
            batch_id=range(N_BATCH)
        )

rule create_federation_endpoints:
    #input: f"{WORK_DIR}/virtuoso-batch-endpoints-ok.txt"
    input: "{workDir}/virtuoso-data-batch{batch_id}-ok.txt"
    output: "{workDir}/virtuoso-federation-endpoints-batch{batch_id}-ok.txt"
    run:
        SPARQL_CONTAINER_NAME = f"docker-{SPARQL_SERVICE_NAME}-{int(wildcards.batch_id)+1}"
        shell(f'docker-compose -f {SPARQL_COMPOSE_FILE} stop')
        shell(f'docker start {SPARQL_CONTAINER_NAME}')

        while not ping(SPARQL_DEFAULT_ENDPOINT):
            LOGGER.debug(f"Waiting for {SPARQL_DEFAULT_ENDPOINT} to start...")
            time.sleep(1)

        global NET_PORT
        NET_PORT += N_BATCH

        proxy_mapping = {}
        virtuoso_mapping_file = f"{VIRTUOSO_PROXY_MAPPING_FILE_PREFIX}-batch{wildcards.batch_id}{VIRTUOSO_PROXY_MAPPING_FILE_EXT}"

        if os.path.exists(virtuoso_mapping_file):
            with open(virtuoso_mapping_file, "r") as f:
                proxy_mapping = json.load(f)

        # Create the federation endpoints
        federation_members_info = CONFIG_GEN["virtuoso"]["federation_members"][f"batch{wildcards.batch_id}"]
        for fed_member_name, fed_member_iri in federation_members_info.items():
            lpath = f"/{fed_member_name}/sparql"
            proxy_target = f"http://localhost:{NET_PORT}{lpath}"
            host = f"localhost:{NET_PORT}"
            proxy_mapping[fed_member_iri] = proxy_target
            NET_PORT += 1
            
            if USE_DOCKER:
                shell(f"python fedshop/virtuoso.py create-sparql-endpoint --container-name={SPARQL_CONTAINER_NAME} --on-duplicate=REPLACE --lpath={lpath} {host} {fed_member_iri}")
            else:
                shell(f'python fedshop/virtuoso.py create-sparql-endpoint --isql="{VIRTUOSO_PATH_TO_ISQL}" --on-duplicate=REPLACE --lpath={lpath} {host} {fed_member_iri}')

        with open(virtuoso_mapping_file, "w") as f:
            json.dump(proxy_mapping, f)
        validate(str(output))
        
rule ingest_data:
    input:
        datafiles=get_data_files,
        containers_created="{workDir}/virtuoso-containers-ok.txt"
    output: "{workDir}/virtuoso-data-batch{batch_id}-ok.txt"
    run:
        SPARQL_CONTAINER_NAME = f"docker-{SPARQL_SERVICE_NAME}-{int(wildcards.batch_id)+1}"
        if USE_DOCKER:
            shell(f'docker-compose -f {SPARQL_COMPOSE_FILE} stop')
            shell(f'docker start {SPARQL_CONTAINER_NAME}')
            while not ping(SPARQL_DEFAULT_ENDPOINT):
                LOGGER.debug(f"Waiting for {SPARQL_DEFAULT_ENDPOINT} to start...")
                time.sleep(1)
            
            shell(f'python fedshop/virtuoso.py ingest-data --container-name {SPARQL_CONTAINER_NAME} --datafiles "{",".join(input.datafiles)}"')
        else:
            shell(f'python fedshop/virtuoso.py ingest-data --isql "{VIRTUOSO_PATH_TO_ISQL}" --datapath {os.path.realpath(VIRTUOSO_PATH_TO_DATA)} --datafiles "{",".join(input.datafiles)}"')
        
        validate(str(output))

rule create_batches:
    output: "{workDir}/virtuoso-containers-ok.txt"
    run:
        shell(f"docker-compose -f {SPARQL_COMPOSE_FILE} create --no-recreate --scale {SPARQL_SERVICE_NAME}={N_BATCH} && touch {output}")