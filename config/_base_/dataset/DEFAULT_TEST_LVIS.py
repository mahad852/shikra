LVIS_TEST_CONFIG = dict(
    type='LVISDataset',
    image_folder='/datasets/MSCOCO17/',
    template_file=r"{{fileDirname}}/template/lvis.json",
)


DEFAULT_TEST_LVIS = dict(
    LVIS_TEST=dict(
        **LVIS_TEST_CONFIG,
        filename=r'{{fileDirname}}/../../../../data/lvis_ann.jsonl',
    )
)