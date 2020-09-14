"""
This file contains everything to talk to Upstride API

Exceptionally, this file doesn't have unittest, as the functions here depends on the Alchemy platform
Maybe in the future we could add integration tests for testing the communication

"""
import requests
import tensorflow as tf
from . import metrics

server_url = 'https://api.upstride.io'


def _get_header(jwt):
  """ return the javascript header with jwt tocken for api call
  """
  return {'Authorization': "Bearer " + jwt}


def send_metric_callbacks(server_args):
  """create and return a keras callback to send data to the server

  Args:
      server_args (Dict): parsed arguments from argument_parser
  """
  def send_metric(epoch, logs):
    """
    Args:
        logs: dictionary containing  loss, accuracy, top_k_categorical_accuracy, val...

    """
    # prepare message
    json_content = []
    for key in logs:
      json_content.append({'value': float(logs[key]), 'name': key})
    # send it with jwt or basic auth security
    if server_args['jwt'] != '':
      headers = _get_header(server_args['jwt'])
      requests.put(server_url + f'/training/metric/{server_args["id"]}', headers=headers, json={'epoch': epoch, 'values': json_content})
    elif server_args['password'] and server_args['user']:
      auth = (server_args['user'], server_args['password'])
      requests.put(server_url + f'/training/metric/{server_args["id"]}', auth=auth, json={'epoch': epoch, 'values': json_content})
  return tf.keras.callbacks.LambdaCallback(on_epoch_end=send_metric)


def send_final_checkpoint():
  # TODO not yet implemented
  pass


def send_exported_model(args, archive_path):
  auth = (args['server']['user'], args['server']['password'])
  with open(archive_path, 'rb') as f:
    files = {'file_to_upload': f}
    r = requests.post(f'{server_url}/training/upload_saved_model/{args["server"]["id"]}', auth=auth, files=files)


def start_training(args):
  """ send experiment data to the server
  """
  if args['server']['user'] and not args['server']['id']:
    message = {'args': {}}
    for key in args:
      if key not in ['server', 'description', 'title']:
        # we don't want to set these parameters in the training parameters
        message['args'][key] = args[key]
    message['description'] = args['description']
    message['title'] = args['title']
    message['git_tag'] = ''
    try:
      r = requests.post(server_url + '/training', json=message, auth=(args['server']['user'], args['server']['password']), timeout=10)
    except Exception as e:
      print("can't send result to the server, training continue")
    if r.status_code == 401:
      raise Exception("Provided authorization is not valid")
    args['server']['id'] = r.json()
  return args


def send_model_info(model, server_args):
  """compute the number of variable and number of flops and send them to the server

  Args:
      server_args (Dict): argument to connect to the server
  """
  n_params = metrics.count_trainable_params(model)
  n_flops = metrics.count_flops_efficient(model)
  print("number of parameters:", n_params)
  print("number of flops:", n_flops)
  message = {
      'n_flops': n_flops,
      'n_params': n_params
  }
  r = requests.put(server_url + f'/training/model_info/{server_args["id"]}', json=message, auth=(server_args['user'], server_args['password']))
