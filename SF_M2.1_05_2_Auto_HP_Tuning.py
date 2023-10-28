from clearml import Task, Logger
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, 
    RandomSearch, UniformIntegerParameterRange)


task = Task.init(project_name='SF_M2.1_05_2_Auto_HP_Tuning',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)
args = {
    'template_task_id': "84eeae5979e54a379c58dd06863ded29",
    'run_as_service': False,
}

an_optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
    UniformIntegerParameterRange('General/n_train', min_value=50, max_value=200, step_size=50),
    UniformIntegerParameterRange('General/n_test', min_value=500, max_value=2000, step_size=500),
    UniformIntegerParameterRange('General/max_depth', min_value=1, max_value=10, step_size=1),
    UniformIntegerParameterRange('General/random_state', min_value=17, max_value=17, step_size=1),
   ],
   objective_metric_title='accuracy',
   objective_metric_series='MSE',
   objective_metric_sign='min',
)

an_optimizer.start()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()