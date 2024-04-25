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
BENCH_DIR = f"{WORK_DIR}/benchmark/generation"
TEMPLATE_DIR = f"{MODEL_DIR}/watdiv"

CONFIG = load_config(CONFIGFILE)
CONFIG_GEN = CONFIG["generation"]
CONFIG_EVAL = CONFIG["evaluation"]

USE_DOCKER = CONFIG["use_docker"]

SPARQL_COMPOSE_FILE = CONFIG_GEN["virtuoso"]["compose_file"]
VIRTUOSO_COMPOSE_CONFIG = load_config(SPARQL_COMPOSE_FILE)
VIRTUOSO_PROXY_MAPPING_FILE = CONFIG_GEN["virtuoso"]["proxy_mapping"]

SPARQL_SERVICE_NAME = CONFIG_GEN["virtuoso"]["service_name"]
SPARQL_CONTAINER_NAME = VIRTUOSO_COMPOSE_CONFIG["services"][SPARQL_SERVICE_NAME]["container_name"]

SPARQL_BATCH0_SERVICE_NAME = CONFIG_GEN["virtuoso"]["service_name_batch0"]
SPARQL_BATCH0_CONTAINER_NAME = VIRTUOSO_COMPOSE_CONFIG["services"][SPARQL_BATCH0_SERVICE_NAME]["container_name"]

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

def ping(host, port):
    try:
        LOGGER.debug(f"Pinging {host}:{port}...")
        req = requests.get(f"http://{host}:{port}/sparql")
        LOGGER.debug(f"{req}")
        return req.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

#=================
# PIPELINE
#=================

rule all:
    input: f"{WORK_DIR}/virtuoso-federation-endpoints-ok.txt"

rule create_federation_endpoints:
    input: f"{WORK_DIR}/virtuoso-batch-endpoints-ok.txt"
    output: f"{WORK_DIR}/virtuoso-federation-endpoints-ok.txt"
    params:
        virtuoso_mapping_file=VIRTUOSO_PROXY_MAPPING_FILE
    run:
        global NET_PORT
        NET_PORT += N_BATCH

        proxy_mapping = {}

        if os.path.exists(params.virtuoso_mapping_file):
            with open(params.virtuoso_mapping_file, "r") as f:
                proxy_mapping = json.load(f)

        # Create the federation endpoints
        federation_members_info = CONFIG_GEN["virtuoso"]["federation_members"][f"batch{N_BATCH-1}"]
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

        with open(params.virtuoso_mapping_file, "w") as f:
            json.dump(proxy_mapping, f)
        validate(str(output))

rule create_batch_endpoints:
    input: f"{WORK_DIR}/virtuoso-data-ok.txt"
    output: f"{WORK_DIR}/virtuoso-batch-endpoints-ok.txt"
    params:
        virtuoso_mapping_file=VIRTUOSO_PROXY_MAPPING_FILE
    run:
        global NET_PORT

        proxy_mapping = {}

        if os.path.exists(params.virtuoso_mapping_file):
            with open(params.virtuoso_mapping_file, "r") as f:
                proxy_mapping = json.load(f)
         
        federation_members_info = CONFIG_GEN["virtuoso"]["federation_members"]
        for batch_id, federation_members in federation_members_info.items():
            group_iri = f"http://www.{batch_id}.fr/"
            lpath = f"/{batch_id}/sparql"
            proxy_target = f"http://localhost:{NET_PORT}{lpath}"
            proxy_mapping[group_iri] = proxy_target
            host = f"localhost:{NET_PORT}"
            NET_PORT += 1

            # Create a group for each batch
            if USE_DOCKER:
                shell(f"python fedshop/virtuoso.py create-graph-group --container-name={SPARQL_CONTAINER_NAME} --drop-first {group_iri}")
                shell(f"python fedshop/virtuoso.py create-sparql-endpoint --container-name={SPARQL_CONTAINER_NAME} --lpath={lpath} {host} {group_iri}")
            else:
                shell(f'python fedshop/virtuoso.py create-graph-group --isql="{VIRTUOSO_PATH_TO_ISQL}" --drop-first {group_iri}')
                shell(f'python fedshop/virtuoso.py create-sparql-endpoint --isql="{VIRTUOSO_PATH_TO_ISQL}" --lpath={lpath} {host} {group_iri}')

            # Add the members to the group
            for fed_member_name, fed_member_iri in federation_members.items():
                if USE_DOCKER:
                    shell(f"python fedshop/virtuoso.py update-graph-group INS --graph-group={group_iri} --member-iri={fed_member_iri} --container-name={SPARQL_CONTAINER_NAME}")
                else:
                    shell(f'python fedshop/virtuoso.py update-graph-group INS --graph-group={group_iri} --member-iri={fed_member_iri} --isql="{VIRTUOSO_PATH_TO_ISQL}"')

        with open(params.virtuoso_mapping_file, "w") as f:
            json.dump(proxy_mapping, f)
        validate(str(output))
        
rule ingest_data:
    input:
        datafiles=CONFIG_GEN["datafiles"],
        batch0_ingested=f"{WORK_DIR}/virtuoso-data-batch0-ok.txt"
    output: f"{WORK_DIR}/virtuoso-data-ok.txt"
    run:
        if USE_DOCKER:
            shell(f'docker stop {SPARQL_BATCH0_CONTAINER_NAME}')
            shell(f'docker start {SPARQL_CONTAINER_NAME}')
            while not ping("localhost", 8890):
                LOGGER.debug("Waiting for Virtuoso to start...")
                time.sleep(1)
            shell(f'python fedshop/virtuoso.py ingest-data --container-name {SPARQL_CONTAINER_NAME}')
        else:
            shell(f'python fedshop/virtuoso.py ingest-data --isql "{VIRTUOSO_PATH_TO_ISQL}" --datapath {os.path.realpath(VIRTUOSO_PATH_TO_DATA)}')
        
        validate(str(output))

rule ingest_batch0:
    input:
        datafiles=CONFIG_GEN["datafiles_batch0"]
    output: f"{WORK_DIR}/virtuoso-data-batch0-ok.txt"
    run:
        if USE_DOCKER:
            shell(f'docker stop {SPARQL_CONTAINER_NAME}')
            shell(f'docker start {SPARQL_BATCH0_CONTAINER_NAME}')
            while not ping("localhost", 8891):
                LOGGER.debug("Waiting for Virtuoso to start...")
                time.sleep(1)

            shell(f'python fedshop/virtuoso.py ingest-data --container-name {SPARQL_BATCH0_CONTAINER_NAME} --datafiles "{",".join(CONFIG_GEN["datafiles_batch0"])}"')
        else:
            # TODO: kill the current Virtuoso process then relaunch with the batch0 configfile
            shell(f'python fedshop/virtuoso.py ingest-data --isql "{VIRTUOSO_PATH_TO_ISQL}" --datapath {os.path.realpath(VIRTUOSO_PATH_TO_DATA)} --datafiles "{",".join(CONFIG_GEN["datafiles_batch0"])}"')
        
        validate(str(output))