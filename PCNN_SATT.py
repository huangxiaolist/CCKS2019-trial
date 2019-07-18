import tensorflow as tf
from helper import *
import sys;

sys.path.insert(0, './')
from scipy.spatial.distance import cdist


class PCNN(object):
	def __init__(self, params):
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.l2 == 0.0:
			self.regularizer = None
		else:
			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()
		self.add_placeholders()
		nn_out, self.accuracy = self.add_model()
		self.loss = self.add_loss(nn_out)
		self.logits = tf.nn.softmax(nn_out)
		self.train_op = self.add_optimizer(self.loss)

		tf.summary.scalar('acc', self.accuracy)
		tf.summary.scalar('loss', self.loss)
		self.merged_sum = tf.summary.merge_all()
		self.summ_writer = None

	def load_data(self):
		data = pickle.load(open(self.p.dataset, 'rb'))
		self.voc2id = data['voc2id']
		self.id2voc = data['id2voc']
		self.max_pos = data['max_pos']
		self.num_class = len(data['rel2id'])

		# Get word list
		self.wrd_list = list(self.voc2id.items())
		self.wrd_list, _ = zip(*self.wrd_list)

		self.data = data
		print(
			'Document count [{}]:{},  [{}]: {}'.format('train', len(self.data['train']), 'dev', len(self.data['dev'])))

	def fit(self, sess):
		self.summ_writer = tf.summary.FileWriter('tf_board/{}'.format(self.p.name), sess.graph)
		saver = tf.train.Saver(max_to_keep=10)
		save_dir = 'checkpoints/{}/'.format(self.p.name)
		make_dir(save_dir)
		res_dir = 'results/{}/'.format(self.p.name)
		make_dir(res_dir)

		if self.p.restore:
			saver.restore(sess, save_path=save_dir + self.p.name +"-"+ str(self.p.restore_epoch))

		if not self.p.only_eval:
			self.min_train_loss = 1024.0
			for epoch in range(self.p.max_epochs):
				train_loss, train_acc = self.run_epoch(sess, self.data['train'], epoch)
				print('[Epoch {}]: Training Loss: {:.5}, Training acc: {:.5}\n'.format(epoch, train_loss, train_acc))

				if train_loss < self.min_train_loss:
					self.min_train_loss = train_loss
					saver.save(sess, save_path=save_dir + self.p.name, global_step=epoch + 1)
					if epoch % 2 == 0:

						# Evaluate on Test
						saver.restore(sess, save_path=save_dir + self.p.name +'-'+ str(epoch + 1))
						dev_loss, dev_acc, y, y_pred, logit_list, y_hot = self.predict(sess, self.data['dev'])
						dev_prec, dev_rec, dev_f1 = self.calc_prec_recall_f1(y, y_pred, 0)

						y_true = np.array([e[1:] for e in y_hot]).reshape((-1))
						y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
						area_pr = average_precision_score(y_true, y_scores)
						with open('./eval_by_epoch2.txt', 'a') as f:
							f.write('drop: {}\tlr: {}\tfilters:{}\tPrec:{}\tRec :{}\tF1:{}\tArea:{}\tacc:{}'.format(self.p.dropout, self.p.lr, self.p.num_filters, dev_prec, dev_rec, dev_f1, area_pr, dev_acc))

							# pickle.dump({'logit_list': logit_list, 'y_hot': y_hot}, open('results/{}/'.format(self.p.name)), 'wb')

	def add_placeholders(self):

		self.input_x = tf.placeholder(tf.int32, [None, self.p.max_len], name='input_data')
		self.input_y = tf.placeholder(tf.int32, [None, None], name='input_labels')
		self.input_pos1 = tf.placeholder(tf.int32, [None, None], name='input_pos1')
		self.input_pos2 = tf.placeholder(tf.int32, [None, None], name='input_pos2')
		self.input_sub_pos = tf.placeholder(tf.int32, [None, None], name='input_sub_pos')
		self.input_obj_pos = tf.placeholder(tf.int32, [None, None], name='input_obj_pos')
		self.total_bags = tf.placeholder(tf.int32, shape=(), name='total_bags')  # Total number of bags in a batch
		self.total_sents = tf.placeholder(tf.int32, shape=(),
										  name='total_sents')  # Total number of sentences in a batch
		self.dropout = tf.placeholder_with_default(self.p.dropout, shape=(), name='dropout')
		self.sent_num = tf.placeholder(tf.int32, [None, 3], name='sent_num')
		self.mask = tf.placeholder(tf.int32, [None, self.p.max_len], name='mask')

	def add_model(self):
		in_wrds, in_pos1, in_pos2 = self.input_x, self.input_pos1, self.input_pos2

		with tf.variable_scope('Embeddings') as scope:
			embed_init = getEmbeddings(self.wrd_list, self.p.embed_dim)
			_wrd_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=False,
											  regularizer=self.regularizer)
			wrd_unk = tf.zeros([1, self.p.embed_dim])
			wrd_blank = tf.zeros([1, self.p.embed_dim])
			wrd_embeddings = tf.concat([wrd_blank, _wrd_embeddings, wrd_unk], axis=0)

			pos1_embeddings = tf.get_variable('pos1_embeddings', shape=[self.max_pos, self.p.pos_dim],
											  initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
											  regularizer=self.regularizer)
			pos2_embeddings = tf.get_variable('pos2_embeddings', shape=[self.max_pos, self.p.pos_dim],
											  initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
											  regularizer=self.regularizer)

			wrd_embed = tf.nn.embedding_lookup(wrd_embeddings, in_wrds)
			pos1_embed = tf.nn.embedding_lookup(pos1_embeddings, in_pos1)
			pos2_embed = tf.nn.embedding_lookup(pos2_embeddings, in_pos2)
			embeds = tf.concat([wrd_embed, pos1_embed, pos2_embed], axis=-1)

		with tf.variable_scope('PCNN') as scope:
			conv_in_dim = self.p.embed_dim + 2 * self.p.pos_dim
			conv_in = tf.expand_dims(embeds, axis=3)
			print(conv_in.get_shape())
			padding = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
			conv_in = tf.pad(conv_in, padding)

			kernel = tf.get_variable(name='kernel', shape=[3, conv_in_dim, 1, self.p.num_filters],
									 initializer=tf.truncated_normal_initializer(),
									 regularizer=self.regularizer)
			biases = tf.get_variable(name='biases', shape=[self.p.num_filters],
									 initializer=tf.random_normal_initializer(), regularizer=self.regularizer)

			conv = tf.nn.conv2d(conv_in, kernel, strides=[1, 1, 1, 1],
								padding='VALID')  # [sent_num, max_len, 1, 200]

			convRes = tf.nn.relu(conv + biases, name=scope.name)


			# Get piece_cnn
			mask_embeddings = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
			pcnn_mask = tf.nn.embedding_lookup(mask_embeddings, self.mask)
			convRes = tf.reshape(convRes, [-1, self.p.max_len, self.p.num_filters, 1])

			convRes = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, self.p.max_len, 3]) * tf.transpose(convRes, [0, 2, 1,3]), axis=2)
			convRes = tf.reshape(convRes, [-1, self.p.num_filters* 3])
			sent_reps = convRes
			de_out_dim = 3 * self.p.num_filters

		with tf.variable_scope('Sentence_attention') as scope:
			sent_atten_q = tf.get_variable('sent_atten_q', [de_out_dim, 1],
										   initializer=tf.contrib.layers.xavier_initializer())

			def getSentAtten(num):
				num_sents = num[1] - num[0]
				bag_sents = sent_reps[num[0]: num[1]]

				sent_atten_wts = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(bag_sents), sent_atten_q), [num_sents]))

				bag_rep = tf.reshape(tf.matmul(tf.reshape(sent_atten_wts, [1, num_sents]), bag_sents), [de_out_dim])

				return bag_rep

			bag_rep = tf.map_fn(getSentAtten, self.sent_num, dtype=tf.float32)
			bag_rep = tf.layers.dropout(bag_rep, rate=self.p.dropout)

		with tf.variable_scope('FC1') as scope:
			w_rel = tf.get_variable('w_rel', [de_out_dim, self.num_class], initializer=tf.contrib.layers.xavier_initializer(),
									regularizer=self.regularizer)
			b_rel = tf.get_variable('b_rel', initializer=np.zeros([self.num_class]).astype(np.float32), regularizer=self.regularizer)
			nn_out = tf.nn.xw_plus_b(bag_rep, w_rel, b_rel)

		with tf.variable_scope('Accuracy') as scope:
			prob = tf.nn.softmax(nn_out)
			y_pred = tf.argmax(prob, axis=1)
			y_actual = tf.argmax(self.input_y, axis=1)
			accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))

			return nn_out, accuracy

	def add_loss(self, nn_out):
		with tf.name_scope('Loss_op'):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_out, labels=self.input_y))
			if self.regularizer is not None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			return loss

	def add_optimizer(self, loss):
		with tf.name_scope('optimizer'):
			if self.p.opt == 'adam' and not self.p.restore:
				optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:
				optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op = optimizer.minimize(loss)
		return train_op

	def run_epoch(self, sess, data, epoch, shuffle=True):
		losses, accuracies = [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			summary_str, loss, accuracy, _ = sess.run([self.merged_sum, self.loss, self.accuracy, self.train_op], feed_dict=feed)

			losses.append(loss)
			accuracies.append(accuracy)

			bag_cnt += len(batch['sent_num'])

			if step % 10 == 0:
				print('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt,
																								len(self.data['train']),
																								np.mean(
																									accuracies) * 100,
																								np.mean(losses),
																								self.p.name,
																								self.min_train_loss))
				self.summ_writer.add_summary(summary_str, epoch * len(self.data['train']) + bag_cnt)

		accuracy = np.mean(accuracies) * 100
		print("Training Loss:{}, Training acc: {}".format(np.mean(losses), accuracy))
		return np.mean(losses), accuracy

	def predict(self, sess, data, shuffle=True, label='Evaluation on Dev'):
		losses, accuracies, y_pred, y, logit_list, y_actual_hot = [], [], [], [], [], []
		bag_cnt = 0
		for step, batch in enumerate(self.getBatches(data, shuffle)):
			loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict=self.create_feed_dict(batch, split='dev'))
			losses.append(loss)
			accuracies.append(accuracy)

			pred_ind = logits.argmax(axis=1)
			logit_list += logits.tolist()
			y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
			y_pred  += pred_ind.tolist()
			y += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
			bag_cnt += len(batch['sent_num'])
			if step % 100 == 0:
				print('{} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['dev']),
																		np.mean(accuracies) * 100, np.mean(losses),
																		self.p.name))

		print('Dev Accuracy: {}'.format(np.mean(accuracies) * 100))

		return np.mean(losses), np.mean(accuracies) * 100, y, y_pred, logit_list, y_actual_hot

	def create_feed_dict(self, batch, split='train'):
		"""
		Creates a feed dictionary for the batch

		Parameters
		----------
		batch:		contains a batch of bags
		wLabels:	Whether batch contains labels or not
		split:		Indicates the split of the data - train/valid/test

		Returns
		-------
		feed_dict	Feed dictionary to be fed during sess.run
		"""
		X, Y, pos1, pos2, sent_num = batch['X'], batch['Y'], batch['Pos1'], batch['Pos2'], batch['sent_num']
		batch_sub_pos, batch_obj_pos = batch['subpos'], batch['objpos']
		total_sents = len(batch['X'])
		total_bags  = len(batch['Y'])
		x_pad, x_len, pos1_pad, pos2_pad, seq_len, sent_mask = self.pad_dynamic(X, pos1, pos2, batch_sub_pos, batch_obj_pos)

		y_hot = self.getOneHot(Y, 		self.num_class)
		# -1 because NA cannot be in proby

		feed_dict = {}
		feed_dict[self.input_x] 		= np.array(x_pad)
		feed_dict[self.input_pos1]		= np.array(pos1_pad)
		feed_dict[self.input_pos2]		= np.array(pos2_pad)
		feed_dict[self.input_y] = y_hot
		# feed_dict[self.x_len] 			= np.array(x_len)
		# feed_dict[self.seq_len]			= seq_len
		feed_dict[self.total_sents]		= total_sents
		feed_dict[self.total_bags]		= total_bags
		feed_dict[self.sent_num]		= sent_num
		feed_dict[self.mask] = sent_mask
		if split != 'train':
			feed_dict[self.dropout]     = 1.0

		else:
			feed_dict[self.dropout]     = self.p.dropout


		return feed_dict


	def getBatches(self, data, shuffle=True):
		if shuffle: random.shuffle(data)

		for chunk in getChunks(data, self.p.batch_size):
			batch = ddict(list)


			num = 0
			for i, bag in enumerate(chunk):

				batch['X']    	   += bag['X']
				batch['Pos1'] 	   += bag['Pos1']
				batch['Pos2'] 	   += bag['Pos2']
				batch['Y'].append(bag['Y'])
				old_num  = num
				num 	+= len(bag['X'])

				batch['sent_num'].append([old_num, num, i])
				batch['subpos'] += bag['SubPos']
				batch['objpos'] += bag['ObjPos']
			yield batch

	def getOneHot(self, data, num_class, isprob=False):
		"""
		Generates the one-hot representation

		Parameters
		----------
		data:		Batch to be padded
		num_class:	Total number of relations

		Returns
		-------
		One-hot representation of batch
		"""
		temp = np.zeros((len(data), num_class), np.int32)
		for i, ele in enumerate(data):
			for rel in ele:
				rel = int(rel)
				if isprob:	temp[i, rel-1] = 1
				else:		temp[i, rel]   = 1

		return temp


	def pad_dynamic(self, X, pos1, pos2, sub_pos, obj_pos):
		"""
		Pads each batch during runtime.

		Parameters
		----------
		X:		For each sentence in the bag, list of words
		pos1:		For each sentence in the bag, list position of words with respect to subject
		pos2:		For each sentence in the bag, list position of words with respect to object
		sub_type:	For each sentence in the bag, Entity type information of the subject
		obj_type:	For each sentence in the bag, Entity type information of the object
		rel_aliaes:	For each sentence in the bag, Relation Alias information

		Returns
		-------
		x_pad		Padded words
		x_len		Number of sentences in each sentence,
		pos1_pad	Padded position 1
		pos2_pad	Padded position 2
		seq_len 	Maximum sentence length in the batch
		subtype 	Padded subject entity type information
		subtype_len 	Number of entity type for each bag in the batch
		objtype 	Padded objective entity type information
		objtype_len 	Number of entity type for each bag in the batch
		rel_alias_ind 	Padded relation aliasses for each bag in the batch
		rel_alias_len	Number of relation aliases
		"""
		seq_len = 0


		x_len = np.zeros((len(X)), np.int32)


		for i, x in enumerate(X):
			seq_len  = max(seq_len, len(x))
			x_len[i] = len(x)

		x_pad,  _ 	 = self.padData(X, self.p.max_len)
		pos1_pad,  _ 	 = self.padData(pos1, self.p.max_len)
		pos2_pad,  _ 	 = self.padData(pos2, self.p.max_len)
		sent_mask= self.maskData(X, sub_pos, obj_pos, self.p.max_len)

		return x_pad, x_len, pos1_pad, pos2_pad, seq_len, sent_mask

	def padData(self, data, max_len):
		"""
		Pads the data in a batch | Used as a helper function by pad_dynamic

		Parameters
		----------
		data:		batch to be padded
		seq_len:	maximum number of words in the batch

		Returns
		-------
		Padded data and mask
		"""
		pad_data = np.zeros((len(data), max_len), np.int32)
		mask     = np.zeros((len(data), max_len), np.float32)

		for i, ele in enumerate(data):
			pad_data[i, :len(ele)] = ele[:max_len]
			mask    [i, :len(ele)] = np.ones(len(ele[:max_len]), np.float32)

		return pad_data, mask

	def maskData(self,data, sub_pos, obj_pos, max_len):
		mask_data = np.ones((len(data), max_len), np.int32) * 3
		for i, ele in enumerate(data):
			for j, wrd in enumerate(ele):
				if j <= sub_pos[i]:
					mask_data[i, j] = 1
				elif j <= obj_pos[i]:
					mask_data[i, j] = 2
				elif j > obj_pos[i]:
					mask_data[i, j] = 3
		return mask_data

	def calc_prec_recall_f1(self, y_actual, y_pred, none_id):
		"""
		Calculates precision recall and F1 score

		Parameters
		----------
		y_actual:	Actual labels
		y_pred:		Predicted labels
		none_id:	Identifier used for denoting NA relation

		Returns
		-------
		precision:	Overall precision
		recall:		Overall recall
		f1:		Overall f1
		"""
		pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0



		for i in range(len(y_pred)):
			if y_pred[i] != none_id:
				pos_pred += 1.0					# classified as pos example (Is-A-Relation)
				if y_pred[i] == y_actual[i]:
					true_pos += 1.0

		pos_gt = true_pos
		for i in range(len(y_actual)):
			if y_pred[i] == none_id and y_actual[i] != none_id:
				pos_gt += 1.0
		precision 	= true_pos / (pos_pred + self.p.eps)
		recall 		= true_pos / (pos_gt + self.p.eps)
		f1 		= 2 * precision * recall / (precision + recall + self.p.eps)

		return precision, recall, f1

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description='Improving Distantly-Supervised Neural Relation Extraction using Side Information')

	parser.add_argument('-data', dest="dataset", required=True, help='Dataset to use')
	parser.add_argument('-gpu', dest="gpu", default='0', help='GPU to use')
	parser.add_argument('-nGate', dest="wGate", action='store_false', help='Include edgewise-gating in GCN')

	parser.add_argument('-lstm_dim', dest="lstm_dim", default=192, type=int, help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-pos_dim', dest="pos_dim", default=5, type=int, help='Dimension of positional embeddings')
	parser.add_argument('-type_dim', dest="type_dim", default=50, type=int, help='Type dimension')
	parser.add_argument('-alias_dim', dest="alias_dim", default=32, type=int, help='Alias dimension')
	parser.add_argument('-de_dim', dest="de_gcn_dim", default=16, type=int,
						help='Hidden state dimension of GCN over dependency tree')

	parser.add_argument('-de_layer', dest="de_layers", default=1, type=int,
						help='Number of layers in GCN over dependency tree')
	parser.add_argument('-drop', dest="dropout", default=0.8, type=float, help='Dropout for full connected layer')
	parser.add_argument('-rdrop', dest="rec_dropout", default=0.8, type=float, help='Recurrent dropout for LSTM')

	parser.add_argument('-lr', dest="lr", default=0.001, type=float, help='Learning rate')
	parser.add_argument('-l2', dest="l2", default=0.001, type=float, help='L2 regularization')
	parser.add_argument('-epoch', dest="max_epochs", default=50, type=int, help='Max epochs')
	parser.add_argument('-batch', dest="batch_size", default=32, type=int, help='Batch size')
	parser.add_argument('-chunk', dest="chunk_size", default=1000, type=int, help='Chunk size')
	parser.add_argument('-restore', dest="restore", action='store_true',
						help='Restore from the previous best saved model')
	parser.add_argument('-only_eval', dest="only_eval", action='store_true',
						help='Only Evaluate the pretrained model (skip training)')
	parser.add_argument('-opt', dest="opt", default='adam', help='Optimizer to use for training')

	parser.add_argument('-eps', dest="eps", default=0.00000001, type=float, help='Value of epsilon')
	parser.add_argument('-name', dest="name", default='test_' + str(uuid.uuid4()), help='Name of the run')
	parser.add_argument('-seed', dest="seed", default=1234, type=int, help='Seed for randomization')
	parser.add_argument('-logdir', dest="log_dir", default='./log/', help='Log directory')
	parser.add_argument('-config', dest="config_dir", default='./config/', help='Config directory')
	parser.add_argument('-embed_loc', dest="embed_loc", default='./data/vec.txt', help='Log directory')
	parser.add_argument('-embed_dim', dest="embed_dim", default=50, type=int, help='Dimension of embedding')
	parser.add_argument('-n_filters', dest='num_filters', default=200, type=int, help='Filter size of model')
	parser.add_argument('-max_len', dest='max_len', default=60, type=int,
						help='The sentence length will all padded to [default is 60]')

	parser.add_argument('-restore_epoch', dest='restore_epoch', default=0, type=int,
						help='The sentence length will all padded to [default is 60]')
	args = parser.parse_args()

	if not args.restore: args.name = args.name

	# Set gpu to use
	set_gpu(args.gpu)

	# Set seed
	tf.set_random_seed(seed=args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	model = PCNN(args)

	config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		# drop  lr   num_filters
		# params = ['{} {} {}'.format(drop, lr, filters) for drop in [0.8, 0.5] for lr in [0.1, 0.01, 0.001] for filters in [200, 230]]
		# for para in params:
		# 	para = para.split()
		# 	args.dropout = float(para[0])
		# 	args.lr = float(para[1])
		# 	args.num_filters = int(para[2])
		model.fit(sess)
