training_arguments_keras_tuner = [
    ['namespace', 'factor', [[str, 'scale', 'log', ''], [float, 'min', 1, ''], [float, 'max', 1, ''], [float, 'step', 0, ''], ]],  # useless as soon as Diff Architecture search is working
    ['namespace', 'batch_size', [[str, 'scale', 'log', ''], [float, 'min', 128, ''], [float, 'max', 128, ''], [float, 'step', 0, ''], ]],
    ['namespace', 'lr', [[str, 'scale', 'log', ''], [float, 'min', 0.00001, ''], [float, 'max', 0.00001, ''], [float, 'step', 0.00001, ''], ]],

    ['namespace', 'optimizer_param', [
        ['List[str]', 'name', ['sgd_nesterov'], 'optimized to be used'],
        ['namespace', 'momentum', [[str, 'scale', 'log', ''], [float, 'min', 0.9, ''], [float, 'max', 0.9, ''], [float, 'step', 0, ''], ]],
    ]],
]
