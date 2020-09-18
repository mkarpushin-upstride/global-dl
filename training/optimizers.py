import math
from typing import List

import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, SGD, Nadam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


optimizer_list = [
    "adadelta",
    "adagrad",
    "adam",
    "adam_amsgrad",
    "sgd",
    "sgd_momentum",
    "sgd_nesterov",
    "nadam",
    "rmsprop",
]

learning_schedule_list = [
    "",
    "exponential_decay",
    "step_decay",
    "step_decay_schedule",
    "polynomial_decay",
    "inverse_time_decay",
    "cosine_decay",
    "lr_reduce_on_plateau"
]

_END_LEARNING_RATE = 0.00001


class ExponentialDecay(object):
  """Applies exponential decay to the learning rate after each epoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (Float): initial learning rate when the tranining starts

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float):
    self.initial_lr = initial_lr

  def __call__(self, epoch: int):
    return tf.maximum(self.initial_lr * tf.pow(1 - 0.1, epoch), _END_LEARNING_RATE)


class StepDecay(object):
  """Applies Step decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      drop_rate (float): drop_rate defines ratio to multiply the learning rate
      drop_after_num_epoch (int): after how many epochs the drop_rate has to be applied

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float, drop_rate=0.5, drop_after_num_epoch=10):
    self.initial_lr = initial_lr
    self.drop_rate = drop_rate
    self.drop_after_num_epoch = drop_after_num_epoch

  def __call__(self, epoch: int):
    return tf.maximum(self.initial_lr * tf.pow(self.drop_rate, tf.floor(epoch / self.drop_after_num_epoch)), _END_LEARNING_RATE)


class StepDecaySchedule(object):
  """Applies Step decay schedule to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      drop_schedule (list[int]): list of integers to specifiy which epochs to reduce the lr
      drop_rate (float): drop_rate defines ratio to multiply the learning rate
      total_epochs (int): the length to which the lr decay should be applied

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float, drop_schedule: List[int], drop_rate=0.5, total_epochs=100):
    self.initial_lr = initial_lr
    self.drop_schedule = list(set(drop_schedule + [total_epochs]))
    self.drop_schedule.sort()
    self.drop_rate = drop_rate
    self.total_epochs = total_epochs
    self.built = False

  def build(self):
    assert lambda x: x > 0 in self.drop_schedule
    assert max(self.drop_schedule) <= self.total_epochs
    self.built = True

  def __call__(self, epoch: int):
    if not self.built:
      self.build()

    self.schedule = []
    for i in range(len(self.drop_schedule)):
      # store the learning rate change based on the drop rate.
      self.schedule.append(max(round(self.initial_lr * math.pow(self.drop_rate, i), 5), _END_LEARNING_RATE))
    index = [epoch <= x for x in self.drop_schedule].index(True)  # get count of true values
    return self.schedule[index]  # pick the respective lr rate


class PolynomialDecay(object):
  """Applies Polynomial decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      power (float): nth degree polynomial to be applied]
      total_epochs (int): the lenght to which the decay should be applied

  returns: float adjusted learning rate. 
  """

  def __init__(self, initial_lr: float, power=5.0, total_epochs=100):
    self.initial_lr = initial_lr
    self.power = power
    self.total_epochs = total_epochs

  def __call__(self, epoch: int):
    return ((self.initial_lr - _END_LEARNING_RATE) *
            tf.pow((1 - epoch / self.total_epochs), self.power)
            ) + _END_LEARNING_RATE


class InverseTimeDecay(object):
  """Applies inverse time decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      decay_rate (float): decay_rate to be multiplied at each epoch]
      decay_step (float): this controls the how steep the decay would be applied]
      staircase (bool): applies integer floor division there by producing non continous decay

  returns: float adjusted learning rate. 
  """

  def __init__(self, initial_lr: float, decay_rate=0.5, decay_step=1.0, staircase=False):
    self.initial_lr = initial_lr
    self.decay_rate = decay_rate
    self.decay_step = decay_step
    self.staircase = staircase

  def __call__(self, epoch: int):
    if self.staircase:
      return self.initial_lr / (1 + tf.floor(self.decay_rate * epoch / self.decay_step))
    else:
      return self.initial_lr / (1 + self.decay_rate * epoch / self.decay_step)


class CosineDecay(object):
  """Applies cosine decay to the learning rate after each eppoch. 
  This schedule is called via the keras LearningRateScheduler at the end of the epoch. 

  Args:
      initial_lr (float): initial learning rate when the tranining starts
      alpha (float): controls the intensity of the cosine function applied
      total_epochs (int): the lenght to which the decay should be applied

  returns: float adjusted learning rate and keep the value greater than the _END_LEARNING_RATE 
  """

  def __init__(self, initial_lr: float, alpha=0.001, total_epochs=100):
    self.initial_lr = initial_lr
    self.alpha = alpha
    self.total_epochs = total_epochs

  def __call__(self, epoch: int):
    cosine = 0.5 * (1 + tf.cos((math.pi * epoch) / self.total_epochs))
    decayed = (1 - self.alpha) * cosine + self.alpha
    return tf.maximum(self.initial_lr * decayed, _END_LEARNING_RATE)


def get_lr_scheduler(lr: float, total_epochs: int, lr_params: dict):

  lr_schedule_name = lr_params['strategy'].lower()

  get_lr = {
      "exponential_decay": LearningRateScheduler(ExponentialDecay(lr)),
      "step_decay": LearningRateScheduler(StepDecay(
          lr,
          lr_params['decay_rate'],
          lr_params['drop_after_num_epoch'])),
      "step_decay_schedule": LearningRateScheduler(StepDecaySchedule(
          lr,
          lr_params['drop_schedule'],
          lr_params['decay_rate'],
          total_epochs)),
      "polynomial_decay": LearningRateScheduler(PolynomialDecay(
          lr,
          lr_params['power'],
          total_epochs)),
      "inverse_time_decay": LearningRateScheduler(InverseTimeDecay(
          lr,
          lr_params['decay_rate'],
          lr_params['decay_step'],
          lr_params['staircase'])),
      "cosine_decay": LearningRateScheduler(CosineDecay(
          lr,
          lr_params['alpha'],
          total_epochs)),
      "lr_reduce_on_plateau": ReduceLROnPlateau(
          monitor='val_loss',
          factor=lr_params['decay_rate'],
          patience=lr_params['patience'],
          verbose=0,
          mode='auto',
          min_lr=lr_params['min_lr'])
  }
  return get_lr[lr_schedule_name]


def get_optimizer(optimizer_param: dict):

  optimizer_name = optimizer_param['name'].lower()
  lr = optimizer_param['lr']
  optimizer = {
      'adadelta': Adadelta(lr),
      'adagrad': Adagrad(lr),
      'adam': Adam(lr),
      'adam_amsgrad': Adam(lr, amsgrad=True),
      'sgd': SGD(lr),
      'sgd_momentum': SGD(lr, momentum=optimizer_param['momentum']),
      'sgd_nesterov': SGD(lr, momentum=optimizer_param['momentum'], nesterov=True),
      'nadam': Nadam(lr),
      'rmsprop': RMSprop(lr),
  }

  return optimizer[optimizer_name]
