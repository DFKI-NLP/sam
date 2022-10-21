import glob
import logging
import os
import shutil
from collections import defaultdict
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def read_annotations(directory: str) -> Dict[str, List[str]]:
    ann_file_names = glob.glob(f'{directory}/*.ann')
    logger.info(f'read {len(ann_file_names)} from {directory} ...')
    res = {}
    for file_path in ann_file_names:
        fn = os.path.basename(file_path)
        current_annotations = [
            l.lstrip() for l in open(file_path).readlines()
            if not l.lstrip().startswith("#") and len(l.strip()) > 0
        ]
        current_annotations = [annot[:-1] if annot[-1] == "\n" else annot for annot in current_annotations]
        res[fn] = current_annotations
    return res


def append_suffix_to_labels_and_ids(annotations: List[str], suffix: Optional[str] = None) -> List[str]:
    if suffix is None:
        return annotations
    res = []
    for annot in annotations:
        _id, _remaining = annot.split("\t", maxsplit=1)
        _id += suffix
        parts = _remaining.split("\t", maxsplit=1)
        text = parts[1] if len(parts) > 1 else None
        _label, _targets = parts[0].split(" ", maxsplit=1)
        _label += f'-{suffix}'
        _targets_list = []
        for _target_pair in _targets.split(";"):
            _target_list = []
            for t in _target_pair.split(" "):
                if ":" in t:
                    t += suffix
                _target_list.append(t)
            _targets_list.append(" ".join(_target_list))
        _targets = ";".join(_targets_list)
        annot_new = f'{_id}\t{_label} {_targets}'
        if text is not None:
            annot_new += f'\t{text}'
        #assert annot_new == annot , f'mismatch: "{annot_new}" != "{annot}"'
        res.append(annot_new)
    return res


def add_suffix_and_merge(brat_dir_a: str, brat_dir_b: str, out_dir: str, suffix_a: Optional[str] = None,
                         suffix_b: Optional[str] = None):
    """
    Read both brat directories, brat_dir_a and brat_dir_b, and append to all labels and ids in brat_dir_a the suffix
    suffix_a and to all in brat_dir_b suffix_b. Finally, assuming that the same files exist in brat_dir_a and
    brat_dir_b, concatenate the content of same annotations files and write to out_dir (and also the text files from
    either brat_dir_a or brat_dir_b).

    :param brat_dir_a:
    :param brat_dir_b:
    :param out_dir:
    :param suffix_a:
    :param suffix_b:
    :return:
    """

    annotations_read = {suffix: read_annotations(directory=d) for d, suffix in [(brat_dir_a, suffix_a), (brat_dir_b, suffix_b)]}
    assert set(annotations_read[suffix_b]) == set(annotations_read[suffix_b]), f'file name mismatch'

    annotations_suffixed = defaultdict(list)

    for suffix, annot_dict in annotations_read.items():
        for fn, annots in annot_dict.items():
            suffixed = append_suffix_to_labels_and_ids(annotations=annots, suffix=suffix)
            annotations_suffixed[fn].extend(suffixed)

    if os.path.exists(out_dir):
        logger.warning(f'out_dir={out_dir} already exists, it will be overwritten!')
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for fn, annots in annotations_suffixed.items():
        file_path = os.path.join(out_dir, fn)
        with open(file_path, "w") as f:
            f.writelines((annot + "\n" for annot in annots))

    txt_file_names = glob.glob(f'{brat_dir_a}/*.txt')
    for fn_in in txt_file_names:
        fn_base = os.path.basename(fn_in)
        shutil.copyfile(src=fn_in, dst=os.path.join(out_dir, fn_base))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    add_suffix_and_merge(
        brat_dir_a="experiments/prediction/rel@gold_adus/best_2gxdhtb2_goldonly",
        brat_dir_b="experiments/prediction/rel@predicted_adus/best_2gxdhtb2_predictiononly",
        suffix_a="GOLD",
        suffix_b=None,
        out_dir="experiments/prediction/rel@predicted_adus/best_2gxdhtb2_both"
    )
    logger.info("done")
