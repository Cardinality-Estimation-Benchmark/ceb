import sys
sys.path.append(".")
from query_representation.query import *
from losses.losses import *
from cardinality_estimation.featurizer import *

import glob
import argparse
import random
from sklearn.model_selection import train_test_split

def get_query_fns():
    fns = list(glob.glob(args.query_dir + "/*"))

    train_qfns = []
    test_qfns = []
    val_qfns = []

    for qi,qdir in enumerate(fns):
        template_name = os.path.basename(qdir)
        if args.query_templates != "all":
            query_templates = args.query_templates.split(",")
            if template_name not in query_templates:
                print("skipping template ", template_name)
                continue

        # let's first select all the qfns we are going to load
        qfns = list(glob.glob(qdir+"/*.pkl"))
        qfns.sort()

        if args.num_samples_per_template == -1:
            qfns = qfns
        elif args.num_samples_per_template < len(qfns):
            qfns = qfns[0:args.num_samples_per_template]
        else:
            assert False

        if args.test_diff_templates:
            cur_val_fns = []
            assert False
        else:
            cur_val_fns, qfns = train_test_split(qfns,
                    test_size=1-args.val_size,
                    random_state=args.seed)
            cur_train_fns, cur_test_fns = train_test_split(qfns,
                    test_size=args.test_size,
                    random_state=args.seed)

        train_qfns += cur_train_fns
        val_qfns += cur_val_fns
        test_qfns += cur_test_fns

    return train_qfns, test_qfns, val_qfns

def load_qdata(fns):
    qreps = []
    for qfn in fns:
        qrep = load_qrep(qfn)
        qreps.append(qrep)
    return qreps

def main():
    train_qfns, test_qfns, val_qfns = get_query_fns()
    print("""Selected {} train queries, {} test queries, and {} val queries"""\
            .format(len(train_qfns), len(test_qfns), len(val_qfns)))
    trainqs = load_qdata(train_qfns)
    featurizer = Featurizer(trainqs)
            # db = DB(args.user, args.pwd, args.db_host, args.port,
                    # args.db_name, db_years)

    # can be quite memory intensive to load them all
    # valqs = load_qdata(val_qfns)
    # testqs = load_qdata(test_qfns)

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=str, required=False,
            default="./queries/imdb/")
    parser.add_argument("--port", type=int, required=False,
            default=5432)
    parser.add_argument("--result_dir", type=str, required=False,
            default=None)
    parser.add_argument("--query_templates", type=str, required=False,
            default="all")

    parser.add_argument("--seed", type=int, required=False,
            default=13)
    parser.add_argument("--test_diff_templates", type=int, required=False,
            default=0)
    parser.add_argument("--num_samples_per_template", type=int, required=False,
            default=-1)
    parser.add_argument("--test_size", type=float, required=False,
            default=0.5)
    parser.add_argument("--val_size", type=float, required=False,
            default=0.2)

    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
