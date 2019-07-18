import sys

sys.path.append('./')
from helper import *


def prepare_train_bags():
    print('Constructing training bags...')
    train_data = ddict(lambda: {'rels': ddict(list)})
    sent_relation_train = {}
    with open("data/sent_relation_train.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            sentid_relid = line.strip().split()
            sent_relation_train[sentid_relid[0]] = sentid_relid[1]
    with open('./data/sent_train.txt') as f:
        while True:
            line = f.readline()
            if not line:
                line = f.readline()
                if not line:
                    break

            data = line.strip().split()
            _id = '{}_{}'.format(data[1], data[2])
            train_data[_id]['sub'] = data[1]
            train_data[_id]['obj'] = data[2]

            train_data[_id]['rels'][sent_relation_train[data[0]]].append({
                'sent': data[3:],
            })

        print('Completed {} and total train-bags are {}'.format(
            time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S"),
            len(train_data)))
    count = 0
    with open('./data/train_bags.json', 'w') as f:
        for _id, data in train_data.items():
            for rel, sents in data['rels'].items():

                entry = {}
                entry['sub'] = data['sub']
                entry['obj'] = data['obj']

                entry['sents'] = sents
                entry['rel'] = [rel]

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1
                if count % 10000 == 0: print(
                    'Writing Completed {}, {}'.format(count,
                                                      time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))


def prepare_dev_bags():
    print('Constructing dev bags...')
    dev_data = ddict(lambda: {'rels': ddict(list)})
    sent_relation_dev = {}
    with open("data/sent_relation_dev.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            sentid_relid = line.strip().split()
            sent_relation_dev[sentid_relid[0]] = sentid_relid[1]
    with open('./data/sent_dev.txt') as f:
        while True:
            line = f.readline()
            if not line:
                line = f.readline()
                if not line:
                    break

            data = line.strip().split()
            _id = '{}_{}'.format(data[1], data[2])
            dev_data[_id]['sub'] = data[1]
            dev_data[_id]['obj'] = data[2]

            dev_data[_id]['rels'][sent_relation_dev[data[0]]].append({
                'sent': data[3:],
            })

        print('Completed {} and total dev-bags are {}'.format(
            time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S"),
            len(dev_data)))
    count = 0
    with open('./data/dev_bags.json', 'w') as f:
        for _id, data in dev_data.items():
            for rel, sents in data['rels'].items():

                entry = {}
                entry['sub'] = data['sub']
                entry['obj'] = data['obj']

                entry['sents'] = sents
                entry['rel'] = [rel]

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1
                if count % 10000 == 0: print(
                    'Writing Completed {}, {}'.format(count,
                                                      time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))


def prepare_test_bags():
    print('Constructing test bags...')
    test_data = ddict(lambda: {'sents': []})
    with open('./data/sent_test.txt') as f:
        while True:
            line = f.readline()
            if not line:
                line = f.readline()
                if not line:
                    break
            data = line.strip().split()
            _id = '{}_{}'.format(data[1], data[2])

            test_data[_id]['sub'] = data[1]
            test_data[_id]['obj'] = data[2]
            # test_data[_id]['rels'].add(rel2id.get(data['rel'], rel2id['NA']))

            test_data[_id]['sents'].append({
                'sent': data[3:]
            })

        print('Completed {}, test bags is {}'.format(time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S"),
                                                     len(test_data)))

    count = 0
    with open('./data/test_bags.json', 'w') as f:
        for _id, data in test_data.items():

            entry = {}
            entry['sub'] = data['sub']
            entry['obj'] = data['obj']
            entry['sents'] = data['sents']

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
        print(
            'Writing Completed {}, {}'.format(count, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))


if __name__ == "__main__":
    prepare_train_bags()
    prepare_dev_bags()
    prepare_test_bags()

'''
 
'''