GQA_TEST_COMMON_CFG = dict(
    type='GQADataset',
    image_folder=r'/datasets/GQA/images',
    scene_graph_file=r'{{fileDirname}}/../../../../data/sceneGraphs/val_sceneGraphs.json',
    scene_graph_index=None,
)

# use standard q-a mode
DEFAULT_TEST_GQA_VARIANT = dict(
    GQA_QB_BL_VAL_BALANCED=dict(
        **GQA_TEST_COMMON_CFG, version="qb-bl", template_file=r"{{fileDirname}}/template/VQA.json",
        filename=r'{{fileDirname}}/../../../../data/questions1.2/gqa_val_balanced_questions.jsonl'
    ),
)
