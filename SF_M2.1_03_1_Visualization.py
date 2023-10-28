from clearml import Task, Logger
import numpy as np

task = Task.init(project_name='SF_M2.1_03_1_Visualization', task_name='scalars')
logger = task.get_logger()

X = np.random.randint(100, size=20)

for i in range(len(X)):
    logger.report_scalar("Scalars graph", 'series', value=X[i], iteration=i)