# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import csv
import os
import modeling
import optimization_finetuning as optimization
import tokenization
import tensorflow as tf
import pickle
import tf_metrics
import random
import numpy as np
from data_augmentation import data_augmentation
# from loss import bi_tempered_logistic_loss

flags = tf.flags

FLAGS = flags.FLAGS
##for data_generate.py
flags.DEFINE_integer(
    "perc", 10, "The percentage of the supervised data")

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "task_name", 'ner',
    "The task names in [ner, semi_ner]")
## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "albert_base_zh/albert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "use_unlabel", True,
    "Whether to use the unlabel dataset")

flags.DEFINE_bool(
    "data_aug", True,
    "Data augmentation for the input data")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float("num_unlabel_train_epochs", 2.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float("thres", 0.05,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("iterations_per_hook_output", 1000,
                     "How many steps to make in each estimator call.")

params = {
    'batch_size': FLAGS.train_batch_size,
    'data_type': 'sup'

}

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    # self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file):
    """Reads a BIO data."""
    with open(input_file,encoding='utf-8') as f:
      lines = []
      words = []
      labels = []
      for line in f:
        contends = line.strip()
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        if contends.startswith("-DOCSTART-"):
          words.append('')
          continue
        # if len(contends) == 0 and words[-1] == '。':
        if len(contends) == 0:
          l = ' '.join([label for label in labels if len(label) > 0])
          w = ' '.join([word for word in words if len(word) > 0])
          lines.append([l, w])
          words = []
          labels = []
          continue
        words.append(word)
        labels.append(label)
      return lines

class NerProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "train.txt")), "train"
    )

  def get_dev_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
    )

  def get_test_examples(self,data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "test.txt")), "test")

  def get_unlabel_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "unsuper_train.txt")), "unlabel"
    )


  def get_labels(self):
    # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
    return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

  def _create_example(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text=text, label=label))
    return examples


def write_tokens(tokens,mode):
  if mode=="test":
    path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
    wf = open(path,'a')
    for token in tokens:
      if token!="**NULL**":
        wf.write(token+'\n')
    wf.close()

def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer,mode):
  textlist = example.text.split(' ')
  labellist = example.label.split(' ')
  tokens = []
  labels = []
  for i, word in enumerate(textlist):
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    label_1 = labellist[i]
    for m in range(len(token)):
      if m == 0:
        labels.append(label_1)
      # else:
      #     labels.append("X")
    # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[0:(max_seq_length - 2)]
    labels = labels[0:(max_seq_length - 2)]
  ntokens = []
  segment_ids = []
  label_ids = []
  ntokens.append("[CLS]")
  segment_ids.append(0)
  # append("O") or append("[CLS]") not sure!
  label_ids.append(label_map["[CLS]"])
  for i, token in enumerate(tokens):
    ntokens.append(token)
    segment_ids.append(0)
    label_ids.append(label_map[labels[i]])
  ntokens.append("[SEP]")
  segment_ids.append(0)
  # append("O") or append("[SEP]") not sure!
  label_ids.append(label_map["[SEP]"])
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  input_mask = [1] * len(input_ids)
  #label_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    # we don't concerned about it!
    label_ids.append(0)
    ntokens.append("**NULL**")
    #label_mask.append(0)
  # print(len(input_ids))
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length
  #assert len(label_mask) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_ids=label_ids,
    #label_mask = label_mask
  )
  write_tokens(ntokens,mode)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
  """Convert a set of `InputExample`s to a TFRecord file."""
  label_map = {}
  for (i, label) in enumerate(label_list,1):
    label_map[label] = i
  with open('albert_base_ner_checkpoints/label2id.pkl','wb') as w:
    pickle.dump(label_map,w)

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map,
                                     max_seq_length, tokenizer, mode)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    # features["is_real_example"] = create_int_feature(
    #     [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      # "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # # So cast all int64 to int32.
    # for name in list(example.keys()):
    #   t = example[name]
    #   if t.dtype == tf.int64:
    #     t = tf.to_int32(t)
    #   example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):

  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_sequence_output()

  hidden_size = output_layer.shape[-1].value

  output_weight = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 11])

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_sum(tf.multiply(per_example_loss,(1 - tf.cast(tf.equal(labels,100),tf.float32))) )
    probabilities = tf.nn.softmax(logits, axis=-1)
    predict = tf.argmax(probabilities,axis=-1)
    return (loss, per_example_loss, logits,predict)

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings=False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    # tf.logging.info("*** Features ***")
    # for name in sorted(features.keys()):
    #   tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    # is_real_example = None
    # if "is_real_example" in features:
    #   is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    # else:
    #   is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, predicts) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
      (assignment_map, initialized_variable_names) = \
          modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      if params['data_type'] == 'sup':
          train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

          train_hook_list = []
          train_tensors_log = {'loss': total_loss,
                               'global_step': tf.train.get_global_step()}
          train_hook_list.append(tf.train.LoggingTensorHook(
              tensors=train_tensors_log, every_n_iter=FLAGS.iterations_per_hook_output))
          output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              training_hooks=train_hook_list)
      elif params['data_type'] == 'unsup':
          train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

          train_hook_list = []
          train_tensors_log = {'loss': total_loss,
                               'global_step': tf.train.get_global_step()}
          train_hook_list.append(tf.train.LoggingTensorHook(
              tensors=train_tensors_log, every_n_iter=FLAGS.iterations_per_hook_output))
          output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              training_hooks=train_hook_list)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        # def metric_fn(label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        precision = tf_metrics.precision(label_ids,predictions,11,[2,3,4,5,6,7],average="macro")
        recall = tf_metrics.recall(label_ids,predictions,11,[2,3,4,5,6,7],average="macro")
        f = tf_metrics.f1(label_ids,predictions,11,[2,3,4,5,6,7],average="macro")
        #
        return {
          "eval_precision":precision,
          "eval_recall":recall,
          "eval_f": f,
          #"eval_loss": loss,
        }
      eval_metric_ops = metric_fn(per_example_loss, label_ids, logits)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metric_ops)
    else:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions= {'predicts':predicts,
                        'logits':logits,
                        'labels':label_ids})#}predictions can be a tensor or a dict of tensors
    return output_spec

  return model_fn



def input_fn_builder(features, seq_length, is_training, drop_remainder, if_data_aug):
  """Creates an `input_fn` closure to be passed to Estimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_ids)

  def data_aug():
      num_examples = len(all_input_ids)
      num_token = 21128
      id_a = -1
      while True:
          id_a = (id_a + 1)%num_examples
          input_mask = all_input_mask[id_a]
          input_ids = all_input_ids[id_a]
          label_ids = all_label_ids[id_a]
          segment_ids = all_segment_ids[0]
          if FLAGS.data_aug:
              input_ids, input_mask,label_ids = data_augmentation(input_ids, input_mask, label_ids, seq_length, num_token)
          yield {'input_ids':np.array(input_ids,dtype=np.int32),
                'input_mask': np.array(input_mask,dtype=np.int32),
                'label_ids': np.array(label_ids,dtype=np.int32),
              "segment_ids": np.array(segment_ids,dtype=np.int32)
          }

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)
    if if_data_aug:
        d = tf.data.Dataset.from_generator(data_aug,{'input_ids':tf.int32,
                                                  'input_mask':tf.int32,
                                                  'label_ids':tf.int32,
                                                  'segment_ids':tf.int32},
                                                 {'input_ids':tf.TensorShape([seq_length]),
                                                  'input_mask':tf.TensorShape([seq_length]),
                                                  'label_ids':tf.TensorShape([seq_length]),
                                                  'segment_ids':tf.TensorShape([seq_length])})
    else:
        d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list,1):
    label_map[label] = i
  with open('albert_base_ner_checkpoints/label2id.pkl','wb') as w:
    pickle.dump(label_map,w)
  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map,
                                     max_seq_length, tokenizer,None)
    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "ner": NerProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  processor = processors[FLAGS.task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None

  # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  # Cloud TPU: Invalid TPU configuration, ensure ClusterResolver is passed to tpu.
  print("###tpu_cluster_resolver:",tpu_cluster_resolver)
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      log_step_count_steps=FLAGS.iterations_per_hook_output)


  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    # prepare the data here
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    print("###length of total train_examples:",len(train_examples))
    if FLAGS.use_unlabel:
        unlabel_train_examples = processor.get_unlabel_examples(FLAGS.data_dir)
        print("###length of total unlabel_examples:",len(unlabel_train_examples))
        num_unlabel_train_steps = int(len(unlabel_train_examples)/ FLAGS.train_batch_size * FLAGS.num_unlabel_train_epochs)
    else:
        num_unlabel_train_steps = 0
    num_train_steps = int(len(train_examples)/ FLAGS.train_batch_size * FLAGS.num_train_epochs)#TODO: change the num_train_steps
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list) + 1,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps + num_unlabel_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_one_hot_embeddings=False)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.Estimator(model_fn=model_fn,
      config=run_config,params = params)


  if FLAGS.do_train:
    # train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # train_file_exists=os.path.exists(train_file)
    # print("###train_file_exists:", train_file_exists," ;train_file:",train_file)
    # if not train_file_exists: # if tf_record file not exist, convert from raw text file. # TODO
    #     file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    # train_input_fn = file_based_input_fn_builder(
    #     input_file=train_file,
    #     seq_length=FLAGS.max_seq_length,
    #     is_training=True,
    #     drop_remainder=True)
    train_features = convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer)

    train_input_fn = input_fn_builder(features=train_features,
                     seq_length=FLAGS.max_seq_length,
                     is_training=True,
                     drop_remainder=True,
                     if_data_aug=FLAGS.data_aug)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.use_unlabel:
        auged_logits = []
        unlabel_train_examples = unlabel_train_examples[:100]
        unlabel_train_features = convert_examples_to_features(unlabel_train_examples, label_list, FLAGS.max_seq_length,
                                                              tokenizer)
        unlabel_train_input_fn = input_fn_builder(features=unlabel_train_features,
                                                  seq_length=FLAGS.max_seq_length,
                                                  is_training=False,
                                                  drop_remainder=False,
                                                  if_data_aug=FLAGS.data_aug)
        result = estimator.predict(unlabel_train_input_fn)
        K = 3
        for aug_times_id in range(K):
            print(aug_times_id,' times predicting')
            tag = [1]*len(unlabel_train_examples)
            del_num = 0
            print('predicting the Pseudo-Labelling')
            auged_predict_logits = []
            for i,feature in enumerate(unlabel_train_features):
                predict_result = next(result)
                predict_feature = predict_result['predicts']
                predict_logits = predict_result['logits']
                aug_label = predict_result['labels']
                conf = (np.exp(predict_logits).T / (np.sum(np.exp(predict_logits), 1))).T
                conf = conf[:np.sum(feature.input_mask)].max(1).mean()
                # print(i,np.sum(feature.label_ids != predict_feature)/np.sum(feature.input_mask),conf)
                # if np.sum(feature.label_ids != predict_feature)/np.sum(feature.input_mask) > (FLAGS.thres):
                if conf < (1-FLAGS.thres):#0.01
                    tag[i] = 0
                # unlabel_train_features[i].label_ids = predict_feature.tolist()
                # predict_feature = predict_feature.tolist()
                # predict_features.append([label for j,label in enumerate(predict_feature) if (aug_label[j]!=11)])
                auged_predict_logits.append(np.array([logit for j, logit in enumerate(predict_logits) if (aug_label[j] != 100)]))

            auged_logits.append(auged_predict_logits)

        temp_unlabel_train_features = np.array(auged_logits).mean(0).argmax(-1)

        logits = np.array(auged_logits).mean(0)

        for temp in range(100):
            prob = (np.exp(logits[temp]).T / (np.sum(np.exp(logits[temp]), 1))).T
            conf = prob[:np.sum(unlabel_train_features[temp].input_mask)].max(1).mean()
            vars = prob[:np.sum(unlabel_train_features[temp].input_mask)].var(1).mean()
            print(temp, conf,vars,
                  np.sum(temp_unlabel_train_features[temp] != np.array(unlabel_train_features[temp].label_ids)[:-5]))

        for i in range(len(unlabel_train_features)):
            unlabel_train_features[i].label_ids = temp_unlabel_train_features[i].tolist() + [0]*5
        # for temp in range(100):
        #     print(np.argmax((unlabel_train_features_1[temp]) != unlabel_train_features[temp].label_ids) / (
        #             np.sum(unlabel_train_features[temp].input_mask) - 5))

        # for i in range(len(unlabel_train_examples)):
        #     if not tag[i]:
        #         unlabel_train_examples.pop(i-del_num)
        #         del_num += 1
        # print('remain',sum(tag)/len(tag)*100,'%')
        unlabel_train_features = unlabel_train_features + train_features * 5
        random.shuffle(unlabel_train_features)
        unlabel_train_input_fn = input_fn_builder(features=unlabel_train_features,
                         seq_length=FLAGS.max_seq_length,
                         is_training=True,
                         drop_remainder=True,
                         if_data_aug=False)
        estimator.train(input_fn=unlabel_train_input_fn, steps=num_unlabel_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    #######################################################################################################################
    # evaluate all checkpoints; you can use the checkpoint with the best dev accuarcy
    steps_and_files = []
    filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
    for filename in filenames:
        if filename.endswith(".index"):
            ckpt_name = filename[:-6]
            cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
            global_step = int(cur_filename.split("-")[-1])
            if global_step == 0:
                tf.logging.info("Not add {} to eval list.".format(cur_filename))
                continue
            tf.logging.info("Add {} to eval list.".format(cur_filename))
            steps_and_files.append([global_step, cur_filename])
    steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

    output_eval_file = os.path.join(FLAGS.data_dir, "eval_results_albert_zh.txt")
    print("output_eval_file:",output_eval_file)
    tf.logging.info("output_eval_file:"+output_eval_file)
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)

            tf.logging.info("***** Eval results %s *****" % (filename))
            writer.write("***** Eval results %s *****\n" % (filename))
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    #######################################################################################################################

    #result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    #
    #output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    #with tf.gfile.GFile(output_eval_file, "w") as writer:
    #  tf.logging.info("***** Eval results *****")
    #  for key in sorted(result.keys()):
    #    tf.logging.info("  %s = %s", key, str(result[key]))
    #    writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
    with open('albert_base_ner_checkpoints/label2id.pkl','rb') as rf:
      label2id = pickle.load(rf)
      id2label = {value:key for key,value in label2id.items()}
    if os.path.exists(token_path):
      os.remove(token_path)
    predict_examples = processor.get_test_examples(FLAGS.data_dir)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file,mode="test")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    predict_input_fn = file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)
    output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
    with open(output_predict_file,'w') as writer:
      for prediction in result:
        output_line = "\n".join(id2label[id] for id in prediction['predicts'] if id!=0) + "\n"
        writer.write(output_line)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()