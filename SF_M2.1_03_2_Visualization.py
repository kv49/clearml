from clearml import Task, Logger
import numpy as np

task = Task.init(project_name='SF_M2.1_03_2_Visualization', task_name='scatter2d')
logger = task.get_logger()

scatter2d = np.hstack((np.atleast_2d(np.arange(0,20)).T, np.random.randint(40, size=(20, 1))))

logger.report_scatter2d(
    'example_scatter',
    'series_lines+markers',
    scatter=scatter2d,
    xaxis='title x',
    yaxis='title y',
    mode='lines+markers'
    )