from utils import *
from sklearn import svm


def ProcessRawData():
    inputs = LoadImages('../initial/train/*.jpg')
    labels = pd.read_csv('../initial/train.csv').Label.values

    num_samples = inputs.shape[0]
    concated_data = np.concatenate((inputs.reshape([num_samples, -1]), labels.reshape([num_samples, -1])), 1)

    mapping = {
        '1': concated_data[concated_data[:, -1] == 1],
        '2': concated_data[concated_data[:, -1] == 2],
        '3': concated_data[concated_data[:, -1] == 3],
        '4': concated_data[concated_data[:, -1] == 4],
        '5': concated_data[concated_data[:, -1] == 5],
        '6': concated_data[concated_data[:, -1] == 6],
        '7': concated_data[concated_data[:, -1] == 7],
        '8': concated_data[concated_data[:, -1] == 8],
    }

    return mapping


def ShuffleSplit(mapping, train_fraction=0.9):
    train_dic = {}
    valid_dic = {}

    for (key, data) in mapping.items():
        num_samples = data.shape[0]
        num_train = int(num_samples * train_fraction)

        np.random.shuffle(data)

        train_dic[key] = data[:num_train, :]
        valid_dic[key] = data[num_train:, :]

    train = np.concatenate(train_dic.values(), 0)
    valid = np.concatenate(valid_dic.values(), 0)

    np.random.shuffle(train)
    np.random.shuffle(valid)

    train_input = train[:, :-1]
    train_label = train[:, -1]
    valid_input = valid[:, :-1]
    valid_label = valid[:, -1]

    shuffle_mapping = {
        'train_input': train_input,
        'train_label': train_label,
        'valid_input': valid_input,
        'valid_label': valid_label,
    }

    return shuffle_mapping


def mysvm(data):

    input_train = data['train_input']
    input_label = data['train_label']
    valid_input = data['valid_input']
    valid_label = data['valid_label']

    clf = svm.SVC()
    clf.fit(input_train, input_label)
    prediction = clf.predict(valid_input)
    print prediction

def main():
    ''' Pre-process data, only run once '''
    #preprocess_mapping = ProcessRawData()
    #Save('../data/preprocess_data', preprocess_mapping)

    ''' Shuffle pre-processed data and split into train and valid '''
    #mapping = Load('../data/preprocess_data.npz')
    #new_mapping = ShuffleSplit(mapping)
    #Save('../data/shuffle_split', new_mapping)

    data = Load('../data/shuffle_split.npz')
    mysvm(data)

if __name__ == '__main__':
    main()
