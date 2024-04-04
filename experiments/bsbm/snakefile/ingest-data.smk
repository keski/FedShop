import json

import numpy as np
import os
from pathlib import Path
import re

import sys
smk_directory = os.path.abspath(workflow.basedir)
sys.path.append(os.path.join(Path(smk_directory).parent.parent.parent, "fedshop"))

from utils import ping, LOGGER, load_config
from itertools import product

CONFIGFILE = config["configfile"]

USE_DOCKER = False

WORK_DIR = "experiments/bsbm"
QUERY_DIR = f"{WORK_DIR}/queries"
MODEL_DIR = f"{WORK_DIR}/model"
BENCH_DIR = f"{WORK_DIR}/benchmark/generation"
TEMPLATE_DIR = f"{MODEL_DIR}/watdiv"

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
PROXY_PORT = re.search(r":(\d+)", PROXY_SERVER).group(1)
PROXY_SPARQL_ENDPOINT = PROXY_SERVER + "sparql"

# Uncomment if you want to use docker
# VIRTUOSO_PATH_TO_ISQL = "/opt/virtuoso-opensource/bin/isql"
# VIRTUOSO_PATH_TO_DATA = "/usr/share/proj/" 

# Uncomment if you want to use local docker
VIRTUOSO_PATH_TO_ISQL = "/Applications/Virtuoso Open Source Edition v7.2.app/Contents/virtuoso-opensource/bin/isql"
VIRTUOSO_PATH_TO_DATA = f"{MODEL_DIR}/dataset" 

N_QUERY_INSTANCES = CONFIG_GEN["n_query_instances"]
VERBOSE = CONFIG_GEN["verbose"]
N_BATCH = CONFIG_GEN["n_batch"]

# Config per batch
N_VENDOR=CONFIG_GEN["schema"]["vendor"]["params"]["vendor_n"]
N_RATINGSITE=CONFIG_GEN["schema"]["ratingsite"]["params"]["ratingsite_n"]

global NET_PORT
NET_PORT = 34200

#=================
# USEFUL FUNCTIONS
#=================
def validate(filename):
    with open(filename, "w") as f:
        f.write("ok")

#=================
# PIPELINE
#=================

rule all:
    input: f"{WORK_DIR}/virtuoso-federation-endpoints-ok.txt"

rule create_federation_endpoints:
    input: f"{WORK_DIR}/virtuoso-batch-endpoints-ok.txt"
    output: f"{WORK_DIR}/virtuoso-federation-endpoints-ok.txt"
    params:
        virtuoso_mapping_file=f"{WORK_DIR}/virtuoso-proxy-mapping.json"
    run:
        global NET_PORT

        proxy_mapping = {}

        if os.path.exists(params.virtuoso_mapping_file):
            with open(params.virtuoso_mapping_file, "r") as f:
               proxy_mapping = json.load(f)

        for fed_member_name, fed_member_id in product(["vendor"], range(N_VENDOR)):
            proxy_target = f"http://localhost:{NET_PORT}"
            proxy_target_alias = f"{fed_member_name}{fed_member_id}"
            graph_uri = f"http://www.{proxy_target_alias}.fr/"
            host = f"localhost:{NET_PORT}"
            NET_PORT += 1
            if USE_DOCKER:
                shell(f"python fedshop/virtuoso.py create-sparql-endpoint --container-name={SPARQL_CONTAINER_NAME} {host} {graph_uri}")
            else:
                shell(f'python fedshop/virtuoso.py create-sparql-endpoint --isql="{VIRTUOSO_PATH_TO_ISQL}" {host} {graph_uri}')

            # Map the endpoint to a path in the proxy: proxy/vendor0 -> http://www.vendor0.fr
            proxy_mapping[proxy_target_alias] = proxy_target
            shell(f'curl -X GET "{PROXY_SERVER}mapping/set-destination?proxyFrom={proxy_target_alias}&proxyTo={proxy_target}"')

        for fed_member_name, fed_member_id in product(["ratingsite"], range(N_RATINGSITE)):
            proxy_target = f"http://localhost:{NET_PORT}"
            proxy_target_alias = f"{fed_member_name}{fed_member_id}"
            graph_uri = f"http://www.{proxy_target_alias}.fr/"
            host = f"localhost:{NET_PORT}"
            NET_PORT += 1
            if USE_DOCKER:
                shell(f"python fedshop/virtuoso.py create-sparql-endpoint --container-name={SPARQL_CONTAINER_NAME} {host} {graph_uri}")
            else:
                shell(f'python fedshop/virtuoso.py create-sparql-endpoint --isql="{VIRTUOSO_PATH_TO_ISQL}" {host} {graph_uri}')

            # Map the endpoint to a path in the proxy: proxy/vendor0 -> http://www.vendor0.fr
            proxy_mapping[proxy_target_alias] = proxy_target
            shell(f'curl -X GET "{PROXY_SERVER}mapping/set-destination?proxyFrom={proxy_target_alias}&proxyTo={proxy_target}"')

        # Save the mapping
        with open(params.virtuoso_mapping_file, "w") as f:
            json.dump(proxy_mapping, f)
        validate(str(output))

rule create_batch_endpoints:
    input: f"{WORK_DIR}/virtuoso-data-ok.txt"
    output: f"{WORK_DIR}/virtuoso-batch-endpoints-ok.txt"
    params:
        virtuoso_mapping_file=f"{WORK_DIR}/virtuoso-proxy-mapping.json"
    run:
        global NET_PORT

        proxy_mapping = {}

        if os.path.exists(params.virtuoso_mapping_file):
            with open(params.virtuoso_mapping_file, "r") as f:
               proxy_mapping = json.load(f)

        # Create one graph group per batch
        for batch_id in range(N_BATCH):
            proxy_target = f"http://localhost:{NET_PORT}"
            proxy_target_alias = f"batch{batch_id}"
            host = f"localhost:{NET_PORT}"
            NET_PORT += 1
            if USE_DOCKER:
                shell(f"python fedshop/virtuoso.py create-graph-group --container-name={SPARQL_CONTAINER_NAME} --drop-first http://www.batch{batch_id}.fr/")
                shell(f"python fedshop/virtuoso.py create-sparql-endpoint --container-name={SPARQL_CONTAINER_NAME} {host} http://www.batch{batch_id}.fr/")
            else:
                shell(f'python fedshop/virtuoso.py create-graph-group --isql="{VIRTUOSO_PATH_TO_ISQL}" --drop-first http://www.batch{batch_id}.fr/')
                shell(f'python fedshop/virtuoso.py create-sparql-endpoint --isql="{VIRTUOSO_PATH_TO_ISQL}" {host} http://www.batch{batch_id}.fr/')

            # Map the endpoint to a path in the proxy: proxy/batch0 -> http://www.batch0.fr
            proxy_mapping[proxy_target_alias] = proxy_target
            shell(f'curl -X GET "{PROXY_SERVER}mapping/set-destination?proxyFrom={proxy_target_alias}&proxyTo={proxy_target}"')

        # Save the mapping
        with open(params.virtuoso_mapping_file, "w") as f:
            json.dump(proxy_mapping, f)

        # Assign corresponding graphs to each group
        _, vendor_edges = np.histogram(np.arange(N_VENDOR), N_BATCH)
        vendor_edges = vendor_edges[1:].astype(int) + 1

        _, ratingsite_edges = np.histogram(np.arange(N_RATINGSITE), N_BATCH)
        ratingsite_edges = ratingsite_edges[1:].astype(int) + 1

        for batch_id in range(N_BATCH):
            group_iri = f"http://www.batch{batch_id}.fr/"
            for fed_member_name, fed_member_id in product(["vendor"], range(vendor_edges[batch_id])):
                member_iri = f"http://www.{fed_member_name}{fed_member_id}.fr/"
                if USE_DOCKER:
                    shell(f"python fedshop/virtuoso.py update-graph-group INS --graph-group={group_iri} --member-iri={member_iri} --container-name={SPARQL_CONTAINER_NAME}")
                else:
                    shell(f'python fedshop/virtuoso.py update-graph-group INS --graph-group={group_iri} --member-iri={member_iri} --isql="{VIRTUOSO_PATH_TO_ISQL}"')
            
            for fed_member_name, fed_member_id in product(["ratingsite"], range(ratingsite_edges[batch_id])):
                member_iri = f"http://www.{fed_member_name}{fed_member_id}.fr/"
                if USE_DOCKER:
                    shell(f"python fedshop/virtuoso.py update-graph-group INS --graph-group={group_iri} --member-iri={member_iri} --container-name={SPARQL_CONTAINER_NAME}")
                else:
                    shell(f'python fedshop/virtuoso.py update-graph-group INS --graph-group={group_iri} --member-iri={member_iri} --isql="{VIRTUOSO_PATH_TO_ISQL}"')
        
        validate(str(output))

rule ingest_data:
    input:
        vendor=expand("{modelDir}/dataset/vendor{vendor_id}.nq", vendor_id=range(N_VENDOR), modelDir=MODEL_DIR),
        ratingsite=expand("{modelDir}/dataset/ratingsite{ratingsite_id}.nq", ratingsite_id=range(N_RATINGSITE), modelDir=MODEL_DIR)
    output: f"{WORK_DIR}/virtuoso-data-ok.txt"
    run:
        if USE_DOCKER:
            shell(f'python fedshop/virtuoso.py ingest-data --container-name {SPARQL_CONTAINER_NAME}')
        else:
            shell(f'python fedshop/virtuoso.py ingest-data --isql "{VIRTUOSO_PATH_TO_ISQL}" --datapath {os.path.realpath(VIRTUOSO_PATH_TO_DATA)}')
        
        validate(str(output))