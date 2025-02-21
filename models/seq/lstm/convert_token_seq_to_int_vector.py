# read vocab
from csv import reader
vocab = {}
with open('vocab.txt') as f:
    words = f.read().splitlines()
    for wordIndex in words:
        word, index = wordIndex.split(' -----> ')
        vocab[word] = index

with open('token_seq_50_test.txt') as f:
    with open('int_seq_50_test.txt', 'w' ,newline='\n') as output:
        codeFiles =  f.read().splitlines()
        for codeFile in codeFiles:
            tokens = codeFile[2:-5].strip().split(', ')
            for token in tokens:
                tokenClean  = token.replace('"', '')
                if tokenClean in vocab:
                    output.write(vocab[tokenClean] + ',')
                else:
                    output.write(vocab['<unk_token>'] + ',')
            output.write('\n')