import numpy as np
import argparse
import time

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
    print("look back: ", seq_len)
    model = load_model('/home/vishalt11/model_with_new_tokens_100_window.hd5')
    score = []
    keys_saved = []
    keys_saved_ratio = []
    # top k suggestions
    k = 1

    #Cnt = 0

    # read the file with tokenised version of codes
    with open(int_seq_fp) as fp:
        for line in fp:
            # get the ids
            seq = [int(x) for x in line.split(',')[:-1]]
            print(len(seq))
            encoded_text_c1 = []
            encoded_text_c2 = []
            seq_word = []
            sub_score = 0
            # get the word form of the ids
            for key in seq:
                seq_word.append(idToWord[key])
            # total character count for the sequence
            # temp = []
            # for word in seq_word:
            #     temp.append(len(word))

            # nc_seq_word = sum(temp)
            pred_word = []
            count = 0
            for idx in range(0, len(seq) - 1):
                # handles the cases with length less than 1000
                if (seq_len - idx) > 0:
                    encoded_text_c1.append(seq[idx])
                    value = int(vocab['<pad_token>'])
                    pad_encoded = pad_sequences([encoded_text_c1], maxlen=seq_len, truncating='pre', value=value)

                # handle the cases above 1000
                else:
                    for jdx in range(seq_len - 1, -1, -1):
                        encoded_text_c2.append(seq[idx - jdx])
                    pad_encoded = pad_sequences([encoded_text_c2], maxlen=seq_len, truncating='pre', value=value)

                # total number of predictions

                top_k = []
                top_k_id = []

                #start = time.process_time()
                # append the top k suggestions in words and ids
                for token_id in (model.predict(pad_encoded)[0]).argsort()[-k:][::-1]:
                    top_k.append(idToWord[token_id])
                    top_k_id.append(token_id)
                #print("Top ", k, " Suggestions: ", top_k)
                #stop = time.process_time()
                #print(stop - start)
                # ground to be compared with the prediction
                next_token = seq[idx + 1]

                #print("ground truth", idToWord[next_token])


                if next_token != 1000:
                    count += 1
                    if next_token in top_k_id:
                        #pred_word.append(idToWord[next_token])
                        # total number of successful predictions
                        sub_score += 1

            # character count for the predicted words
            # temp = []
            # for word in pred_word:
            #     temp.append(len(word))
            # nc_pred_word = sum(temp)

            # ks_ratio = nc_pred_word/nc_seq_word
            #Cnt += count

            score.append(sub_score/count)
            #keys_saved.append(nc_pred_word)
            #keys_saved_ratio.append(ks_ratio)
            #print("Keys saved: ", nc_pred_word, "save ratio: ", ks_ratio)
            print(score, sub_score, count)
            avg_score = sum(score) / len(score)
            print("Mean score: ", avg_score)

    #print("input_seq", Cnt)

    with open('score_out_v1k_newtokens_100lb.txt', 'w') as fp:
        for idx in range(0, len(score)):
            fp.write('%.4f\n' % (score[idx]))
        fp.write('%.4f\n' % (avg_score))

def GetVectors(vocab_fp, int_seq_fp):
    vocab = {}
    idToWord = {}
    with open(vocab_fp) as f:
        words = f.read().splitlines()
        for wordIndex in words:
            word, index = wordIndex.split(' -----> ')
            vocab[word] = index
            idToWord[int(index)] = word

    look_back_len = 100 + 1
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
    int_seq_fp = 'int_seq_50_test.txt'
    n_sequences, vocab, vocabulary_size, idToWord = GetVectors(vocab_fp, int_seq_fp)
    #Train(n_sequences, vocabulary_size)
    Predict(n_sequences, vocab, idToWord, int_seq_fp)

    # parser = argparse.ArgumentParser(description="do stuff")
    # parser.add_argument(
    #     "--input", "-i", help="input code"
    # )
    # args = parser.parse_args()
    # input_seq_word = args.input.split(" ")
    # print(input_seq_word)
    #
    # input_seq_id = []
    # for inp in input_seq_word:
    #     for key, value in vocab.items():
    #         if inp == key:
    #             input_seq_id.append(value)
    #
    # train_inputs = n_sequences[:, :-1]
    # seq_len = train_inputs.shape[1]
    # #seq_len = 5
    # model = load_model('model_9.hd5')
    #
    # value = int(vocab['<pad_token>'])
    # pad_encoded = pad_sequences([input_seq_id], maxlen=seq_len, truncating='pre', value=value)
    #
    # print("top 5 suggestions: ")
    # for token_id in (model.predict(pad_encoded)[0]).argsort()[-5:][::-1]:
    #     print(idToWord[token_id])

if __name__ == "__main__":
    main()