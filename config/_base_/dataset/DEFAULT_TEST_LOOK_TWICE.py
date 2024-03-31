POINT_TEST_COMMON_CFG_LOOK_TWICE = dict(
    type='Point_QA_twice',
    image_folder='/datasets/VG',
    template_file=r"{{fileDirname}}/template/VQA.json",
)


DEFAULT_TEST_LOOK_TWICE = dict(
    POINT_LOOK_TWICE_oq_b_val=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='oq-b', filename='{{fileDirname}}/../../../../data/pointQA_twice_val.jsonl'),
    POINT_LOOK_TWICE_oq_p_val=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='oq-p', filename='{{fileDirname}}/../../../../data/pointQA_twice_val.jsonl'),
    POINT_LOOK_TWICE_sq_b_val=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='sq-b', filename='{{fileDirname}}/../../../../data/pointQA_twice_val.jsonl'),
    POINT_LOOK_TWICE_sq_p_val=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='sq-p', filename='{{fileDirname}}/../../../../data/pointQA_twice_val.jsonl'),
    POINT_LOOK_TWICE_gq_b_val=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='gq-b', filename='{{fileDirname}}/../../../../data/pointQA_twice_val.jsonl'),
    POINT_LOOK_TWICE_gq_p_val=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='gq-p', filename='{{fileDirname}}/../../../../data/pointQA_twice_val.jsonl'),

    POINT_LOOK_TWICE_oq_b_test=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='oq-b', filename='{{fileDirname}}/../../../../data/pointQA_twice_test.jsonl'),
    POINT_LOOK_TWICE_oq_p_test=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='oq-p', filename='{{fileDirname}}/../../../../data/pointQA_twice_test.jsonl'),
    POINT_LOOK_TWICE_sq_b_test=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='sq-b', filename='{{fileDirname}}/../../../../data/pointQA_twice_test.jsonl'),
    POINT_LOOK_TWICE_sq_p_test=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='sq-p', filename='{{fileDirname}}/../../../../data/pointQA_twice_test.jsonl'),
    POINT_LOOK_TWICE_gq_b_test=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='gq-b', filename='{{fileDirname}}/../../../../data/pointQA_twice_test.jsonl'),
    POINT_LOOK_TWICE_gq_p_test=dict(**POINT_TEST_COMMON_CFG_LOOK_TWICE, version='gq-p', filename='{{fileDirname}}/../../../../data/pointQA_twice_test.jsonl'),
)
