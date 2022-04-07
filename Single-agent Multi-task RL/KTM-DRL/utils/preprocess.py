import datetime
import json
import pprint
import os


def load_cfg(json_dir, base_dir, task, seed=0):
    run_id = datetime.datetime.now().strftime('%d%H%M%S')
    with open(json_dir) as fin:
        cfg = json.load(fin)

    cfg["seed"] = seed
    cfg["task"] = task

    cfg["dir"] = {"base": base_dir}
    base = os.path.join(cfg["dir"]["base"], task)
    run_id = "{}-s{}".format(run_id, seed)

    if cfg["flag"] == "eval": run_id += "-eval"

    cfg["dir"]["log"] = os.path.join(base, run_id)
    os.makedirs(cfg["dir"]["log"], exist_ok=True)

    cfg["dir"]["mod"] = os.path.join(base, run_id, 'mod')
    os.makedirs(cfg["dir"]["mod"], exist_ok=True)

    cfg["dir"]["sum"] = os.path.join(base, run_id)
    os.makedirs(cfg["dir"]["sum"], exist_ok=True)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    with open(os.path.join(cfg["dir"]["sum"], "config.json"), 'w') as fout:
        json.dump(cfg, fout, indent=2)

    return cfg
