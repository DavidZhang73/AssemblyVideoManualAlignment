import json
import os
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm

DATASET_PATH = os.path.join("dataset")  # Change the path to your dataset path
DATASET_JSON_PATHNAME = os.path.join(DATASET_PATH, "IkeaAssemblyInstructionDataset.json")


def load_image(pathname):
    return np.array(Image.open(pathname).convert("RGB"))


def load_image_list(pathname_list):
    return np.array([load_image(pathname) for pathname in pathname_list])


def get_furniture_image_list_map():
    """

    :return: {
        "furniture_id": {
            "page_image_list": [],
            "step_image_list": [],
            "step_index_to_page_index": {}
        },
        "max_page_image_list_length": 0,
        "max_step_image_list_length": 0
    }
    """
    ret = dict(max_page_image_list_length=0, max_step_image_list_length=0)
    with open(DATASET_JSON_PATHNAME, "r", encoding="utf8") as f:
        dataset = json.load(f)
    for furniture in tqdm(dataset):
        manual_page_count_list = []
        step_index_to_page_index = {}
        furniture_id = furniture["id"]
        furniture_path = os.path.join(DATASET_PATH, "Furniture", furniture["subCategory"], furniture["id"])

        # page
        manual_path = os.path.join(furniture_path, "manual")
        page_pathname_list = []
        for m in sorted(os.listdir(manual_path)):
            m_path = os.path.join(manual_path, m)
            if os.path.isdir(m_path):
                page_name_list = os.listdir(m_path)
                page_name_list = filter(lambda x: x.endswith("224.png"), page_name_list)
                page_name_list = sorted(page_name_list, key=lambda x: int(x.split(".")[0].split("-")[1]))
                manual_page_count_list.append(len(page_name_list))
                page_pathname_list.extend([os.path.join(m_path, page_name) for page_name in page_name_list])
        page_image_list = load_image_list(page_pathname_list)

        # step
        step_path = os.path.join(furniture_path, "step")
        step_name_list = os.listdir(step_path)
        step_name_list = filter(lambda x: x.endswith("224.png"), step_name_list)
        step_name_list = sorted(step_name_list, key=lambda x: int(x.split(".")[0].split("-")[1]))
        step_pathname_list = [os.path.join(step_path, step_name) for step_name in step_name_list]
        step_image_list = load_image_list(step_pathname_list)

        # step_index_to_page_index
        for annotation in furniture["annotationList"]:
            manual = int(annotation["manual"])
            page = int(annotation["page"])
            step = int(annotation["step"])
            page_count = 0
            for i in range(manual):
                page_count += manual_page_count_list[i]
            page_index = page_count + page
            step_index_to_page_index[step] = page_index
            assert page_index < len(page_image_list)

        ret[furniture_id] = dict(
            page_image_list=page_image_list,
            step_image_list=step_image_list,
            step_index_to_page_index=step_index_to_page_index,
        )
        ret["max_page_image_list_length"] = max(ret["max_page_image_list_length"], len(page_image_list))
        ret["max_step_image_list_length"] = max(ret["max_step_image_list_length"], len(step_image_list))
    return ret


def gen():
    furniture_image_list_map = get_furniture_image_list_map()
    with open(os.path.join(DATASET_PATH, "image.pkl"), "wb") as f:
        pickle.dump(furniture_image_list_map, f)


if __name__ == "__main__":
    gen()
