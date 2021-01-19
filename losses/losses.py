import numpy as np
import pdb
from losses.plan_losses import PPC, PlanCost
from utils.utils import deterministic_hash

import multiprocessing as mp
import random
from collections import defaultdict
import pandas as pd
import networkx as nx
import datetime
import os

def fix_query(query):
    # these conditions were needed due to some edge cases while generating the
    # queries on the movie_info_idx table, but crashes pyscopg2 somewhere.
    # Removing them shouldn't effect the queries.
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in query:
        query = query.replace(bad_str1, "")
    if bad_str2 in query:
        query = query.replace(bad_str2, "")

    return query

def _get_all_cardinalities(queries, preds):
    ytrue = []
    yhat = []
    for i, pred_subsets in enumerate(preds):
        qrep = queries[i]["subset_graph"].nodes()
        keys = list(pred_subsets.keys())
        keys.sort()
        for alias in keys:
            pred = pred_subsets[alias]
            actual = qrep[alias]["cardinality"]["actual"]
            if actual == 0:
                actual += 1
            ytrue.append(float(actual))
            yhat.append(float(pred))
    return np.array(ytrue), np.array(yhat)

def compute_qerror(queries, preds, result_dir=None):
    '''
    @queries: [qrep_1, ...qrep_N]
    @preds: [{},...,{}]

    @ret: [qerror_1, ..., qerror_{num_subplans}]
    Each query has multiple subplans; the returned list flattens it into a
    single array. The subplans of a query are sorted alphabetically (see
    _get_all_cardinalities)
    '''
    assert len(preds) == len(queries)
    assert isinstance(preds[0], dict)

    ytrue, yhat = _get_all_cardinalities(queries, preds)
    assert len(ytrue) == len(yhat)
    assert 0.00 not in ytrue
    assert 0.00 not in yhat

    errors = np.maximum((ytrue / yhat), (yhat / ytrue))

    # TODO: result files

    return errors

def compute_abs_error(queries, preds, result_dir=None):
    ytrue, yhat = _get_all_cardinalities(queries, preds)
    errors = np.abs(yhat - ytrue)
    return errors

def compute_relative_error(queries, preds, result_dir=None):
    '''
    '''
    ytrue, yhat = _get_all_cardinalities(queries, preds)
    # TODO: may want to choose a minimum estimate
    # epsilons = np.array([1]*len(yhat))
    # ytrue = np.maximum(ytrue, epsilons)

    errors = np.abs(ytrue - yhat) / ytrue
    return errors

def compute_postgres_plan_cost(queries, preds,
        user="arthurfleck",pwd="password", db_name="imdb", db_host="localhost",
        port=5432, num_processes=-1, result_dir=None, cost_model="cm1"):
    '''
    @queries: list of qreps.
    @preds: list of dicts; each dict contains estimates for all subplans of
that query. Order of queries in querues and preds should be the same.
    @kwargs:
        # the following are read from ppcironment variables, if set. Otherwise
        # uses the passed in value
        user: default: $LCARD_USER
        port: default: $LCARD_PORT
        ...see default values of others, or pass in appropriate values.
        cost_model: this is just a convenient key to specify the PostgreSQL
        configuration to use. You can implement new versions in the function
        set_cost_model. e.g., cm1: disable materialization and parallelism, and
        enable all other flags.
    @ret:
        pg_costs:
        pg_plans:
        pg_sqls: TODO.
        TODO: decide how to save result logs, incl. sqls to execute.
    '''
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)

    if "LCARD_USER" in os.environ:
        user = os.environ["LCARD_USER"]
    if "LCARD_PORT" in os.environ:
        port = os.environ["LCARD_PORT"]

    if num_processes == -1:
        pool = mp.Pool(int(mp.cpu_count()))
    else:
        pool = mp.Pool(num_processes)

    ppc = PPC(cost_model, user, pwd, db_host,
            port, db_name)

    est_cardinalities = []
    true_cardinalities = []
    sqls = []
    join_graphs = []

    for i, qrep in enumerate(queries):
        sqls.append(qrep["sql"])
        join_graphs.append(qrep["join_graph"])
        ests = {}
        trues = {}
        predq = preds[i]
        for node, node_info in qrep["subset_graph"].nodes().items():
            est_card = predq[node]
            alias_key = ' '.join(node)
            trues[alias_key] = node_info["cardinality"]["actual"]
            if est_card == 0:
                est_card += 1
            ests[alias_key] = est_card
        est_cardinalities.append(ests)
        true_cardinalities.append(trues)

    # some edge cases to handle to get the queries to work in the PostgreSQL
    for i,sql in enumerate(sqls):
        sqls[i] = fix_query(sql)

    costs, opt_costs, plans, sqls = \
                ppc.compute_costs(sqls, join_graphs,
                        true_cardinalities, est_cardinalities,
                        num_processes=num_processes,
                        pool=pool)
    # debug code
    # losses = costs - opt_costs
    # print("num samples: ", len(losses))
    # print("losses mean: ", np.mean(losses))
    # print("costs mean: ", np.mean(costs))
    # print("opt costs mean: ", np.mean(opt_costs))

    # TODO: create log files etc. if appropriate kwargs passed in
    for i, qrep in enumerate(queries):
        sql_key = str(deterministic_hash(qrep["sql"]))
        # TODO: save log files
        # add_query_result_row(sql_key, samples_type,
                # est_sqls[i], est_costs[i],
                # losses[i],
                # get_leading_hint(est_plans[i]),
                # qrep["template_name"], cur_costs, costs,
                # qrep["name"])
    pool.close()
    return costs, opt_costs

def compute_plan_cost(queries, preds, cost_model="C",
        num_processes=-1):
    '''
    @queries: list of qreps.
    @preds: list of dicts; each dict contains estimates for all subplans of
that query. Order of queries in querues and preds should be the same.
    @cost_model: string label for cost model; assigns cost to each edge in the
    plan graph. see cost_model.py for implementation.

    @ret:
        plan_costs:
    '''
    assert isinstance(queries, list)
    assert isinstance(preds, list)
    assert isinstance(queries[0], dict)

    if num_processes == -1:
        pool = mp.Pool(int(mp.cpu_count()))
    else:
        pool = mp.Pool(num_processes)

    pc = PlanCost(cost_model)
    costs, opt_costs = pc.compute_costs(queries, preds, pool=pool)

    pool.close()
    return costs, opt_costs
