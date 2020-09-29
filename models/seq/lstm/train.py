import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

def Train(n_sequences, vocabulary_size):
    train_inputs = n_sequences[:, :-1]
    train_targets = n_sequences[:,-1]

    train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
    seq_len = train_inputs.shape[1]
    print(train_targets[0])
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    print(model.summary())
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_inputs,train_targets,epochs=500,verbose=1)
    model.save("mymodel.h5")

def Predict(n_sequences, vocab, idToWord, int_seq_fp):
    train_inputs = n_sequences[:, :-1]
    seq_len = train_inputs.shape[1]
    model = load_model('mymodel.h5')
    # ratio of successful predicted tokens for each input sequence
    score = []
    # top k suggestions
    k = 5
    with open(int_seq_fp) as fp:
        for line in fp:
            seq = [int(x) for x in line.split(',')[:-1]]
            print(seq)
            encoded_text_c1 = []
            encoded_text_c2 = []
            sub_score = 0
            for idx in range(0, len(seq) - 1):
                to_print = []
                if (seq_len - idx) > 0:
                    encoded_text_c1.append(seq[idx])
                    value = int(vocab['<pad_token>'])
                    pad_encoded = pad_sequences([encoded_text_c1], maxlen=seq_len, truncating='pre', value=value)
                    for x in pad_encoded[0]:
                        to_print.append(idToWord[x])
                    print("Input sequence: ", to_print)
                else:
                    encoded_text_c2 = [seq[idx-4], seq[idx-3], seq[idx-2], seq[idx-1], seq[idx]]
                    pad_encoded = pad_sequences([encoded_text_c2], maxlen=seq_len, truncating='pre', value=value)
                    for x in pad_encoded[0]:
                        to_print.append(idToWord[x])
                    print("Input sequence: ", to_print)

                top_k = []
                top_k_id = []
                for token_id in (model.predict(pad_encoded)[0]).argsort()[-k:][::-1]:
                    #print(idToWord[token_id])
                    top_k.append(idToWord[token_id])
                    top_k_id.append(token_id)

                print("Top ", k, " Suggestions: ", top_k_id)
                next_token = seq[idx + 1]
                print("ground truth", next_token)

                if next_token in top_k_id:
                    sub_score += 1
                print("\n")
            score.append(sub_score/len(seq))
    print("accuracy for prediction for each code file")
    print(score)

    encoded_text1 = [100000, 20, 10875, 17, 100000, 20, 100000, 17, 100000, 20, 100000, 17, 100000, 20, 264, 17, 25,
                     1040, 4, 8, 0, 279, 2, 2515, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0,
                     100000, 2, 100000, 1, 2, 0, 100000, 2, 43897, 1, 2, 0, 100000, 2, 37254, 1, 2, 0, 100000, 2, 100000,
                     1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000,
                     2, 6389, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 35]

    encoded_text2 = [100000, 20, 10875, 17, 100000, 20, 100000, 17, 100000, 20, 100000, 17, 100000, 20, 264, 17, 25,
                     1040, 4, 8, 0, 279, 2, 2515, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0,
                     100000, 2, 100000, 1, 2, 0, 100000, 2, 43897, 1, 2, 0, 100000, 2, 37254, 1, 2, 0, 100000, 2,
                     100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2,
                     0, 100000, 2, 6389, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2,
                     3512, 1, 2, 0, 100000, 2, 3512, 1, 2, 0, 100000, 2, 3512, 1, 2, 0, 100000, 2, 16227, 1, 2, 0,
                     100000, 2, 460, 1, 2, 0, 100000, 2, 43898, 1, 2, 0, 100000, 2, 43898, 1, 2, 0, 100000, 2, 43898, 1,
                     2, 0, 100000, 2, 43898, 1, 2, 0, 100000, 2, 43898, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2,
                     100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2,
                     0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 100000, 2,
                     100000, 1, 2, 0, 100000, 2, 100000, 1, 2, 0, 94887, 2, 2098, 1, 7]

def GetVectors(vocab_fp, int_seq_fp):
    vocab = {}
    idToWord = {}
    with open(vocab_fp) as f:
        words = f.read().splitlines()
        for wordIndex in words:
            word, index = wordIndex.split(' : ')
            vocab[word] = index
            idToWord[int(index)] = word

    look_back_len = 5 + 1
    sequences = []
    vocabulary_size = len(vocab)

    with open(int_seq_fp) as f:
        files = f.read().splitlines()
        for file in files:
            numbers = list(map(int, file.split(',')[:-1]))
            # print(numbers)
            for i in range(look_back_len, len(numbers)):
                seq = numbers[i - look_back_len:i]
                sequences.append(seq)
        # print(sequences)
        n_sequences = np.empty([len(sequences), look_back_len], dtype='int32')
        for i in range(len(sequences)):
            n_sequences[i] = sequences[i]

    return n_sequences, vocab, vocabulary_size, idToWord

def main():
    vocab_fp = 'vocab.txt'
    int_seq_fp = 'int-seq.txt'
    n_sequences, vocab, vocabulary_size, idToWord = GetVectors(vocab_fp, int_seq_fp)
    #Train(n_sequences, vocabulary_size)
    Predict(n_sequences, vocab, idToWord, int_seq_fp)

if __name__ == "__main__":
    main()