import os
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import file_exists, get_file_name
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(desc=f"Downloading '{file_name_with_ext}' to buffer...", total=fsize) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###

    dataset_path = "CelebAMask-HQ"
    img_path = os.path.join(dataset_path, "CelebA-HQ-img")
    mask_dir = os.path.join(dataset_path, "CelebAMask-HQ-mask-anno")
    tag_file = os.path.join(dataset_path, "CelebAMask-HQ-attribute-anno.txt")
    batch_size = 50

    tag_dict = defaultdict()

    with open(tag_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.split()
            if i == 1:
                dict_keys = line
            elif i > 1:
                tag_dict[get_file_name(line[0])] = {
                    key: line[i + 1] for i, key in enumerate(dict_keys)
                }

    classes = [
        "hair",
        "l_brow",
        "l_eye",
        "l_lip",
        "mouth",
        "neck",
        "nose",
        "r_brow",
        "r_eye",
        "skin",
        "u_lip",
        "cloth",
        "l_ear",
        "r_ear",
        "ear_r",
        "hat",
        "neck_l",
        "eye_g",
    ]

    def create_ann(image_path):
        labels = []
        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]
        file_idx = get_file_name(image_path)
        if file_idx in ["11", "85", "116", "2212", "4583"]:
            print(1)
        image_tags = [tag for tag in tag_dict[file_idx] if tag_dict[file_idx][tag] == "1"]
        tags = []
        for tag in image_tags:
            tag = [sly.Tag(tag_meta) for tag_meta in tag_metas if tag_meta.name == tag]
            tags.extend(tag)
        mask_idx = file_idx.rjust(5, "0")
        possible_masks = [f"{mask_idx}_{label}.png" for label in classes]
        for mask_file in possible_masks:
            mask_dir_id = int(file_idx) // 2000  # 2k files in dir
            mask_path = os.path.join(mask_dir, str(mask_dir_id), mask_file)
            if file_exists(mask_path):
                mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
                mask_height = mask_np.shape[0]
                mask_wight = mask_np.shape[1]
                if len(np.unique(mask_np)) != 1:
                    for color in np.unique(mask_np):
                        if color == 0:
                            continue
                        mask = mask_np == 255
                        label_name = mask_file[6 : mask_file.find(".png")]
                        obj_class = meta.get_obj_class(label_name)
                        curr_bitmap = sly.Bitmap(mask)
                        scaled_bitmap = curr_bitmap.resize(
                            (mask_height, mask_wight), (img_height, img_wight)
                        )
                        curr_label = sly.Label(scaled_bitmap, obj_class)
                        labels.append(curr_label)
        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_classes = [sly.ObjClass(name, sly.Bitmap) for name in classes]
    tag_metas = [sly.TagMeta(name, sly.TagValueType.NONE) for name in dict_keys]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=obj_classes, tag_metas=tag_metas)
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, "ds", change_name_if_conflict=True)

    img_names = os.listdir(img_path)
    progress = sly.Progress("Create dataset {}".format("ds"), len(img_names))

    for img_names_batch in sly.batched(img_names, batch_size=batch_size):
        images_pathes_batch = [os.path.join(img_path, image_name) for image_name in img_names_batch]
        img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]
        anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)
        progress.iters_done_report(len(img_names_batch))
