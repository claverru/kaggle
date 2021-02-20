import os

import pandas as pd

dtype_test = {
    'row_id': 'int64', 
    'timestamp': 'int64', 
    'user_id': 'int32', 
    'content_id': 'int16', 
    'content_type_id': 'int8',
    'task_container_id': 'int16', 
    'user_answer': 'int8', 
    'answered_correctly': 'int8', 
    'prior_question_elapsed_time': 'float32', 
    'prior_question_had_explanation': 'boolean',
}


def test_generator(test_dir):
    for root, dirs, files in os.walk(test_dir, topdown=False):
        for name in files:
            test_path = os.path.join(root, name)
            yield pd.read_csv(test_path, dtype=dtype_test, index_col=0)