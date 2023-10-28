from clearml import PipelineDecorator
from clearml import Task, Logger

@PipelineDecorator.component(cache=True, execution_queue="default")
def lib():
    from random import randint
    import numpy as np
    from sklearn.model_selection import train_test_split

@PipelineDecorator.component(return_values=['xs, ys'], execution_queue="default")
def step_one():
    from random import randint
    import numpy as np

    xs = np.linspace(0, 10, 50)
    ys = xs**2 + np.random.random(50) * 10

    return xs, ys

@PipelineDecorator.component(return_values=['xs1'], execution_queue="default")
def step_two(xs: float, t: int = 0):
    import numpy as np
    if t == 0:
        xs1 = np.c_[xs]
    if t == 1:
        xs1 = np.c_[xs, xs**2]
    if t == 2:
        xs1 = np.c_[xs, xs**2, xs**3]
    if t == 3:
        xs1 = np.c_[xs, xs**2, xs**3, xs**4]
    if t == 4:
        xs1 = np.c_[xs, xs**2, xs**3, xs**4, xs**5]          
    return xs1

@PipelineDecorator.component(return_values=['X_train, X_test, y_train, y_test'], execution_queue="default")
def step_three(xs1, ys):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(xs1, ys, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

@PipelineDecorator.component(return_values=['model'], execution_queue="default")
def step_four(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

@PipelineDecorator.component(return_values=['score'], execution_queue="default")
def step_five(model, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    score = model.score(X_test, y_test)
    return score

@PipelineDecorator.pipeline(name='pipe1_pipe1', project='pipe1_pipe1', version='0.0.1')
def executing_pipeline():
    task = Task.init(project_name="pipe1_pipe1", task_name="pipe1_pipe1")
    logger = task.get_logger()
    lib()
    xs, ys = step_one()
    for i in range(5):
        xs1 = step_two(xs, i)
        X_train, X_test, y_train, y_test = step_three(xs1, ys)
        model = step_four(X_train, y_train)
        logger.report_scalar(
            "Metric", "series", iteration=i, value=step_five(model, X_test, y_test)
        )
        print(step_five(model, X_test, y_test))

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    executing_pipeline()
    PipelineDecorator.stop()