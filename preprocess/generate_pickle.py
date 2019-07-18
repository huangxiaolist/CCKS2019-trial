import sys; 
sys.path.append('./')

from helper import *
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from orderedset import OrderedSet

parser = argparse.ArgumentParser(description='Main Preprocessing program')
parser.add_argument('-test', 	 dest="FULL", 		action='store_false')
parser.add_argument('-pos', 	 dest="MAX_POS", 	default=40,   	 	type=int, help='Max position to consider for positional embeddings')
parser.add_argument('-mvoc', 	 dest="MAX_VOCAB", 	default=1380000,  	type=int, help='Maximum vocabulary to consider')
parser.add_argument('-maxw', 	 dest="MAX_WORDS", 	default=60, 	 	type=int)
parser.add_argument('-minw', 	 dest="MIN_WORDS", 	default=5, 	 	type=int)
parser.add_argument('-num', 	 dest="num_procs", 	default=40, 	 	type=int)
parser.add_argument('-thresh', 	 dest="thresh", 	default=0.65, 	 	type=float)
parser.add_argument('-nfinetype',dest='wFineType', 	action='store_false')
parser.add_argument('-metric',   default='cosine')
parser.add_argument('-data', 	 default='riedel')

# Change the below two arguments together
parser.add_argument('-embed',    dest="embed_loc", 	default='./data/vec.txt')
parser.add_argument('-embed_dim',default=50, 		type=int)

# Below arguments can be used for testing processing script (process a part of data instead of full)
parser.add_argument('-sample', 	 dest='FULL', 		action='store_false', 	 		help='To process the entire data or a sample of it')
parser.add_argument('-samp_size',dest='sample_size', 	default=200, 	 	type=int,	help='Sample size to use for testing processing script')
args = parser.parse_args()

print('Starting Data Pre-processing script...')
rel2id        = json.loads(open('./data/relation2id.json').read())
id2rel 	      = dict([(v, k) for k, v in rel2id.items()])

# embed_model   = gensim.models.KeyedVectors.load_word2vec_format(args.embed_loc, binary=False)
#



data = {
	'train': [],
	'dev': [],
	'test':  []
}

def get_index(arr, ele):
	if ele in arr:  return arr.index(ele)
	else:       return -1

def read_file(file_path):
	temp = []

	with open(file_path) as f:
		for k, line in enumerate(f):
			bag   = json.loads(line.strip())
			wrds_list 	= []
			pos1_list 	= []
			pos2_list 	= []
			sub_pos_list   	= []
			obj_pos_list    = []
			for sent in bag['sents']:


				sub_pos, obj_pos = None, None
				dep_links 	 = []

				# Iterating over sentences

				i = 0

				for token in sent['sent']:
					if token == bag['sub']:
						sub_pos = i						# Indexing starts from 0

					elif token == bag['obj']:
						obj_pos = i
					i = i + 1

				if sub_pos == None or obj_pos == None:
					print('Skipped entry!!')
					print('{} | {} | {}'.format(bag['sub'], bag['obj'], sent['sent']))
					pdb.set_trace()
					continue

				wrds    = [' '.join(e) 	for e in sent['sent']]
				pos1	= [i - sub_pos 		for i in range(len(sent['sent']))]					# tok_id = (number of tokens + 1)
				pos2	= [i - obj_pos 		for i in range(len(sent['sent']))]

				wrds_list.append(wrds)
				pos1_list.append(pos1)
				pos2_list.append(pos2)
				sub_pos_list.append(sub_pos)
				obj_pos_list.append(obj_pos)
			temp.append({
				'sub':			bag['sub'],
				'obj':			bag['obj'],
				'rels':			bag['rel'],
				'sub_pos_list':		sub_pos_list,
				'obj_pos_list':		obj_pos_list,
				'wrds_list': 		wrds_list,
				'pos1_list': 		pos1_list,
				'pos2_list': 		pos2_list,
			})
				
			if k % 1000 == 0: print('Completed {}'.format(k))
			if not args.FULL and k > args.sample_size: break
	return temp


print('Reading train bags'); data['train'] = read_file( 'data/train_bags.json')
print('Reading dev bags'); data['dev'] = read_file( 'data/dev_bags.json')
# print('Reading test bags');  data['test']  = read_file( 'data/test_bags.json')

print('Bags processed: Train:{},Dev:{}, Test:{}'.format(len(data['train']),len(data['dev']), len(data['test'])))

"""*************************** REMOVE OUTLIERS **************************"""
del_cnt = 0
for dtype in ['train','dev', 'test']:
	for i in range(len(data[dtype])-1, -1, -1):
		bag = data[dtype][i]
		
		for j in range(len(bag['wrds_list'])-1, -1, -1):
			data[dtype][i]['wrds_list'][j] 		= data[dtype][i]['wrds_list'][j][:args.MAX_WORDS]
			data[dtype][i]['pos1_list'][j] 		= data[dtype][i]['pos1_list'][j][:args.MAX_WORDS]
			data[dtype][i]['pos2_list'][j] 		= data[dtype][i]['pos2_list'][j][:args.MAX_WORDS]
			# data[dtype][i]['dep_links_list'][j] 	= [e for e in data[dtype][i]['dep_links_list'][j] if e[0] < args.MAX_WORDS and e[1] < args.MAX_WORDS]
			# if len(data[dtype][i]['dep_links_list'][j]) == 0:
			# 	del data[dtype][i]['dep_links_list'][j]			# Delete sentences with no dependency links

		if len(data[dtype][i]['wrds_list']) == 0:
			del data[dtype][i]
			del_cnt += 1
			continue

print('Bags deleted {}'.format(del_cnt))

"""*************************** GET PROBABLE RELATIONS **************************"""

train_mega_phr_list = []
for i, bag in enumerate(data['train']):
	train_mega_phr_list.append({
		'bag_index': i,

	})

chunks  = partition(train_mega_phr_list, args.num_procs)

test_mega_phr_list = []
for i, bag in enumerate(data['dev']):
	test_mega_phr_list.append({
		'bag_index': i,

	})

chunks  = partition(test_mega_phr_list, args.num_procs)

"""*************************** FORM VOCABULARY **************************"""
voc_freq = ddict(int)
vocab = []
vocab.append('BLANK')
with open("data/vec.txt") as f:
	while True:
		line = f.readline()
		if not line:
			break
		word = line.split()[0]
		vocab.append(word)


vocab.append('UNK')

"""*************************** WORD 2 ID MAPPING **************************"""
def getIdMap(vals, begin_idx=0):
	ele2id = {}
	for id, ele in enumerate(vals):
		ele2id[ele] = id + begin_idx
	return ele2id


voc2id     = getIdMap(vocab, 0)
id2voc 	   = dict([(v, k) for k,v in voc2id.items()])

# type_vocab = OrderedSet(['NONE'] + list(set(mergeList(ent2type.values()))))
# type2id    = getIdMap(type_vocab)

print('Chosen Vocabulary:\t{}'.format(len(vocab)))
# print('Type Number:\t{}'.format(len(type2id)))

"""******************* CONVERTING DATA IN TENSOR FORM **********************"""


def getId(wrd, wrd2id, def_val='NONE'):
	if wrd in wrd2id: return wrd2id[wrd]
	else: 		  return wrd2id[def_val]


def posMap(pos):
	if   pos < -args.MAX_POS: return 0
	elif pos > args.MAX_POS:  return (args.MAX_POS + 1)*2
	else: 			  return pos + (args.MAX_POS+1)


def procData(data, split='train'):
	res_list = []

	for bag in data:
		res = {}												# Labels will be K - hot
		res['X'] 	 = [[getId(wrd, voc2id, 'UNK')  for wrd in wrds] for wrds in bag['wrds_list']]
		res['Pos1'] 	 = [[posMap(pos) 		for pos in pos1] for pos1 in bag['pos1_list']]
		res['Pos2'] 	 = [[posMap(pos) 		for pos in pos2] for pos2 in bag['pos2_list']]
		res['Y']    	 = bag['rels']
		res['SubPos']    = bag['sub_pos_list']
		res['ObjPos']    = bag['obj_pos_list']


		res_list.append(res)

	return res_list

final_data = {
	'train':  	procData(data['train'], 'train'),
	'dev':		procData(data['dev'], "dev"),
	'test':	  	procData(data['test'],  'test'),
	'voc2id': 	voc2id,
	'id2voc': 	id2voc,
	'rel2id':	rel2id,
	# 'type2id':	type2id,
	'max_pos':	(args.MAX_POS+1)*2 + 1
}

pickle.dump(final_data, open('data/person_processed.pkl', "wb"))