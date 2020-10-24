# read vocab
from csv import reader
vocab = {}
with open('vocab-old.txt') as f:
    words = f.read().splitlines()
    for wordIndex in words:
        word, index = wordIndex.split(' -----> ')
        vocab[word] = index

with open('token-seq-old') as f:
    with open('int-seq-old.txt', 'w' ,newline='\n') as output:
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