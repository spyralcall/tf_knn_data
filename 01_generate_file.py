import os, MeCab

# 分かち書きしたファイル格納用のディレクトリ
if not os.path.exists('wakati_toiawase_claim_meisi'):
    os.mkdir('wakati_toiawase_claim_meisi')

tagger = MeCab.Tagger()
tagger.parseToNode('') # おまじない

def lineWakatiWriter(line, writer):
    node = tagger.parseToNode(line)
    while node:
        if node.feature.startswith('名詞'):
            writer.write(node.surface + '\n')
        node = node.next

for file in os.listdir(''):
    with open('aozora/' + file, 'rt') as reader, open('wakati/' + file, 'wt') as writer:
        for line in reader:
            lineWakatiWriter(line, writer)
