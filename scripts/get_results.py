import os

import jinja2
import numpy as np
import pyrootutils
import wandb

pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=False, pythonpath=True, cwd=True)

WANDB_PROJECT = r"davidz/AssemblyVideoManualAlignment"
WANDB_RUN_NAME_FILTER_REGEX = r"cr"
JINJA2_TEMPLATE_PATH = os.path.join("scripts", "templates")
JINJA2_OUTPUT_PATH = os.path.join("scripts", "output")
JINJA2_DATA_LIST = [
    dict(
        file_name="losses.tex",
        data=None,
    ),
    dict(
        file_name="losses_ot.tex",
        data=None,
    ),
    dict(
        file_name="losses_dtw.tex",
        data=None,
    ),
    dict(
        file_name="result.tex",
        data=None,
    ),
]


def four(x):
    if x < 1:
        return f"{x:#.3g}"
    else:
        return f"{x:#.4g}"


def percent(x):
    return four(x * 100)


JINJA2_DATA_COLUMNS = [
    {"test/accuracy/top1/video/step": dict(format=percent)},
    {"test/accuracy/top1/video/page": dict(format=percent)},
    {"test/index_error/video/step": dict(format=four)},
    {"test/index_error/video/page": dict(format=four)},
    {"test/recall@1/video/step": dict(format=percent)},
    {"test/recall@1/video/page": dict(format=percent)},
    {"test/recall@3/video/step": dict(format=percent)},
    {"test/recall@3/video/page": dict(format=percent)},
    {"test/AUROC/video/step": dict(format=four)},
    {"test/AUROC/video/page": dict(format=four)},
    {"test/kendall/video/step": dict(format=four)},
    {"test/kendall/video/page": dict(format=four)},
    {"test/accuracy/top1/ot/video/step": dict(format=percent)},
    {"test/accuracy/top1/ot/video/page": dict(format=percent)},
    {"test/index_error/ot/video/step": dict(format=four)},
    {"test/index_error/ot/video/page": dict(format=four)},
    {"test/recall@1/ot/video/step": dict(format=percent)},
    {"test/recall@1/ot/video/page": dict(format=percent)},
    {"test/recall@3/ot/video/step": dict(format=percent)},
    {"test/recall@3/ot/video/page": dict(format=percent)},
    {"test/AUROC/ot/video/step": dict(format=four)},
    {"test/AUROC/ot/video/page": dict(format=four)},
    {"test/kendall/ot/video/step": dict(format=four)},
    {"test/kendall/ot/video/page": dict(format=four)},
    {"test/accuracy/top1/dtw/video/step": dict(format=percent)},
    {"test/accuracy/top1/dtw/video/page": dict(format=percent)},
    {"test/index_error/dtw/video/step": dict(format=four)},
    {"test/index_error/dtw/video/page": dict(format=four)},
    {"test/recall@1/dtw/video/step": dict(format=percent)},
    {"test/recall@1/dtw/video/page": dict(format=percent)},
    {"test/recall@3/dtw/video/step": dict(format=percent)},
    {"test/recall@3/dtw/video/page": dict(format=percent)},
    {"test/AUROC/dtw/video/step": dict(format=four)},
    {"test/AUROC/dtw/video/page": dict(format=four)},
    {"test/kendall/dtw/video/step": dict(format=four)},
    {"test/kendall/dtw/video/page": dict(format=four)},
]


def get_data_list():
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, {"display_name": {"$regex": WANDB_RUN_NAME_FILTER_REGEX}})
    data = dict()
    for run in runs:
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        for column in JINJA2_DATA_COLUMNS:
            for key, value in column.items():
                if key in summary:
                    summary[key] = value["format"](summary[key])
        data[run.name] = summary
        data[run.name]["_config"] = config
    for template in JINJA2_DATA_LIST:
        template["data"] = data

    return JINJA2_DATA_LIST


def render(data_list):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(JINJA2_TEMPLATE_PATH),
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\#{",
        comment_end_string="}",
        line_statement_prefix="%-",
        line_comment_prefix="%#",
        trim_blocks=True,
        autoescape=False,
    )
    for data in data_list:
        with open(os.path.join(JINJA2_OUTPUT_PATH, data["file_name"]), "w") as f:
            f.write(env.get_template(data["file_name"]).render(data["data"]))


def main():
    data_list = get_data_list()
    render(data_list)


if __name__ == "__main__":
    main()
    # print(four(25.4))
