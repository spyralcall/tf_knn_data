import os, MeCab

# 分かち書きしたファイル格納用のディレクトリ
if not os.path.exists('train_wakati_toiawase_claim_meisi'):
    os.mkdir('train_wakati_toiawase_claim_meisi')
    
if not os.path.exists('test_wakati_toiawase_claim_meisi'):
    os.mkdir('test_wakati_toiawase_claim_meisi')

tagger = MeCab.Tagger()
tagger.parseToNode('') # おまじない

def lineWakatiWriter(line, writer):
    node = tagger.parseToNode(line)
    while node:
        if node.feature.startswith('名詞'):
            writer.write(node.surface + '\n')
        node = node.next

for file in os.listdir('train'):
    with open('train/' + file, 'rt') as reader, open('train_wakati_toiawase_claim_meisi/' + file, 'wt') as writer:
        for line in reader:
            lineWakatiWriter(line, writer)


for file in os.listdir('test'):
    with open('test/' + file, 'rt') as reader, open('test_wakati_toiawase_claim_meisi/' + file, 'wt') as writer:
        for line in reader:
            lineWakatiWriter(line, writer)
