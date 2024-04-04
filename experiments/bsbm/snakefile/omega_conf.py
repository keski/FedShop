from omegaconf import OmegaConf
import ast
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import time
import colorlog
import numpy as np
import requests
from scipy.stats import norm, truncnorm
from omegaconf import OmegaConf
import psutil
import pandas as pd

# Define resolvers
class RandomBucket:	
    def __init__(self, size):
        self._cumulativePercentage = [None] * size
        self._objects = [None] * size
        self._index=0
        self._totalPercentage = 0.0
	
    def add(self, percentage, obj):
        if self._index == len(self._objects):
            return
        else:
            self._objects[self._index] = obj
            self._cumulativePercentage[self._index] = percentage
            self._totalPercentage += percentage
		
        self._index += 1
		
        if self._index == len(self._objects):
            cumul = 0.0
            for i in range(len(self._objects)):
                cumul += self._cumulativePercentage[i] / self._totalPercentage
                self._cumulativePercentage[i] = cumul

	
    def getRandom(self):
        randIndex = np.random.uniform()
		
        for i in range(len(self._objects)):
            if randIndex <= self._cumulativePercentage[i]:
                return self._objects[i]
		
        # Should never happens, but...
        return self._objects[len(self._objects)-1]

class NormalDistGenerator:
    def __init__(self, mu, sigma, avg) -> None:
        self._avg = avg
        self._mu = mu
        self._sigma = sigma
    
    def getValue(self) -> int:
        randVal: float = norm.ppf(np.random.rand(), loc=self._mu, scale=self._sigma)

        while randVal < 0:
            randVal = norm.ppf(np.random.rand(), loc=self._mu, scale=self._sigma)
        
        return int(((randVal / self._mu) * self._avg) + 1)

class NormalDistRangeGenerator:
    def __init__(self, mu, sigma, maxValue, normalLimit) -> None:
        self._mu = mu
        self._sigma = sigma
        self._maxValue = maxValue
        self._normalLimit = normalLimit
    
    def getValue(self) -> int:
        randVal: float = norm.ppf(np.random.rand(), loc=self._mu, scale=self._sigma)

        while randVal > self._normalLimit or randVal < 0:
            randVal = norm.ppf(np.random.rand(), loc=self._mu, scale=self._sigma)
        
        return int(((randVal / self._normalLimit) * self._maxValue) + 1)

def divide(*args):
    if len(args) != 2:
        raise RuntimeError(f"The number of arguments must not exceed 2. Args: {args}")
    
    return int(args[0] / args[1])

def get_branching_factors(nbProducts):
    """Compute the branching factor given the number of products. Ref: bsbmtool

    Args:
        nbProducts (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    logSF = np.log10(nbProducts)

    # depth = log10(scale factor)/2 + 1
    depth = round(logSF / 2) + 1
		
    branchingFactors = [None] * depth
    branchingFactors[0] = 2 * round(logSF)

    temp = [2, 4, 8]
    for i in range(depth):
        if (i+1) < depth:
            branchingFactors[i] = 8
        else:
            value = temp[round(logSF*3/2+1) % 3]
            branchingFactors[i] = value

    return branchingFactors

def create_product_type_hierarchy(nbProducts):
    branchFt = get_branching_factors(nbProducts)
    oldDepth = -1
    depth = 0
    nr = 1
    
    maxProductTypeNrPerLevel = []
    productTypeLeaves = []
    productTypeNodes = []

    typeQueue = [depth]
    while len(typeQueue) > 0:
        parent_type = typeQueue.pop(0)
        depth = parent_type

        if oldDepth != depth:
            oldDepth = depth
            maxProductTypeNrPerLevel.append(nr)

        for _ in range(branchFt[parent_type]):
            nr += 1
            child_type = parent_type + 1

            if parent_type == len(branchFt)-1:
                productTypeLeaves.append(child_type)
            else:
                productTypeNodes.append(child_type)
                typeQueue.append(child_type)

    
    if nr != maxProductTypeNrPerLevel[len(maxProductTypeNrPerLevel)-1]:
        maxProductTypeNrPerLevel.append(nr)
    
    return productTypeLeaves, productTypeNodes

def get_product_features(nbProducts):
    """Compute the number of features given the number of products, and the random number of required feature for 1 product. Ref: bsbmtool

    Args:
        nbProducts (_type_): _description_

    Returns:
        _type_: _description_
    """
    productTypeLeaves, productTypeNodes = create_product_type_hierarchy(nbProducts)
    leaves_features, nodes_features = [None] * len(productTypeLeaves), [None] * len(productTypeNodes)
    depth = productTypeLeaves[0]
    featureFrom = [None] * depth
    featureTo = [None] * depth

    featureFrom[0] = featureTo[0] = 5
    depthSum = depth * (depth+1) / 2 - 1

    for i in range(2, depth+1):
        featureFrom[i-1] = int(35 * i / depthSum)
        featureTo[i-1] = int(75 * i / depthSum)

    productFeatureNr = 1

    for i, node in enumerate(productTypeNodes):
        if i == 0: continue
        _from = featureFrom[node]
        _to = featureTo[node] + 1

        _count = np.random.randint(_from, _to)
        productFeatureNr += _count
        nodes_features[i] = _count

    for i, node in enumerate(productTypeLeaves):
        _from = featureFrom[node-1]
        _to = featureTo[node-1] + 1

        _count = np.random.randint(_from, _to)
        productFeatureNr += _count
        leaves_features[i] = _count

    return productFeatureNr, np.random.choice(leaves_features).item()

def generate_producer_distribution(productCount):
    productCountGen = NormalDistGenerator(3, 1, 50)
    productNr = 1
    producerOfProduct = [0]
		
    while productNr <= productCount :
        # Now generate Products for this Producer
        hasNrProducts = productCountGen.getValue()
        if productNr+hasNrProducts-1 > productCount:
            hasNrProducts = productCount - productNr + 1
        productNr += hasNrProducts
        producerOfProduct.append(productNr-1)
    return producerOfProduct

def get_product_producers(nbProducts):
    """Compute the number of producers given the number of products and the number of types per product. Ref: bsbmtool

    Args:
        nbProducts (_type_): _description_

    Returns:
        _type_: _description_
    """
    productNr = 1
    producerNr = 1

    producerOfProduct = generate_producer_distribution(nbProducts)
    nbTypes = []
		
    while producerNr < len(producerOfProduct):	
        # Generate Publisher data		
        hasNrProducts = producerOfProduct[producerNr] - producerOfProduct[producerNr-1]
        nbTypes.append(hasNrProducts)
        # createProductsOfProducer(producerNr, productNr, hasNrProducts, productSeedGen)
			
        productNr += hasNrProducts
        producerNr += 1
    
    return producerNr, np.random.choice(nbTypes).item()

def normal_truncated(mu, sigma, lower, upper):
    return int(truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma))

def get_batch_endpoints(n_batch, proxy_endpoint):
    return [proxy_endpoint + f"services/batch{i}/sparql" for i in range(n_batch)]

def get_federation_endpoints(n_vendor, n_ratingsite, proxy_endpoint):
    return [proxy_endpoint + f"services/vendor{i}/sparql" for i in range(n_vendor)] + [proxy_endpoint + f"ratingsite{i}/sparql" for i in range(n_ratingsite)]
    
# Register new resolvers
OmegaConf.register_new_resolver("multiply", lambda *args: np.prod(args).item())
OmegaConf.register_new_resolver("sum", lambda *args: np.sum(args).item())
OmegaConf.register_new_resolver("divide", divide)

OmegaConf.register_new_resolver("get_product_type_n", lambda nbProd: len(create_product_type_hierarchy(nbProd)[0]))
OmegaConf.register_new_resolver("get_product_type_c", lambda nbProd: get_product_producers(nbProd)[1])
OmegaConf.register_new_resolver("get_product_feature_n", lambda nbProd: get_product_features(nbProd)[0])
OmegaConf.register_new_resolver("get_product_feature_c", lambda nbProd: get_product_features(nbProd)[1])
OmegaConf.register_new_resolver("get_product_producer_n", lambda nbProd: get_product_producers(nbProd)[0])

OmegaConf.register_new_resolver("normal_dist", lambda *args: NormalDistGenerator(*args).getValue())
OmegaConf.register_new_resolver("normal_dist_range", lambda *args: NormalDistRangeGenerator(*args).getValue())
OmegaConf.register_new_resolver("normal_truncated", normal_truncated)

OmegaConf.register_new_resolver("get_batch_endpoints", get_batch_endpoints)
OmegaConf.register_new_resolver("get_federation_endpoints", get_federation_endpoints)

def load_config(filename, saveAs=None):
    """Load configuration from a file. By default, attributes are interpolated at access time.

    Args:
        filename ([type]): an input template config file
        saveAs ([type], optional): When specified, interpolate all attributes and persist to a file. Defaults to None, meaning that attributes will be interpolated at access time.

    Returns:
        [type]: [description]
    """
    
    config = OmegaConf.load(filename)
    if saveAs is not None:
        cache_config = None
        try: cache_config = OmegaConf.to_object(config)
        except: cache_config = { k: v for k, v in config.items() if k not in ["virtuoso"]}
        
        with open(saveAs, "w") as tmpfile:
            OmegaConf.save(cache_config, tmpfile)

    return config