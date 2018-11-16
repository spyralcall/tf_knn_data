import os
import json
import pandas as pd

from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import config


if not os.path.exists('result_data_perte_20180910_hukusi_josi_keidou_nuki'):
	os.mkdir('result_data_perte_20180910_hukusi_josi_keidou_nuki')



#辞書の設定
with open(config.SETTINGFILE,"r", encoding="utf-8") as f:
	settings = json.load(f)
tokenizer_obj = dictionary.Dictionary(settings).create()


def wakati_by_sudachi(text,writer):
	"""
	sudachiを使った分かち書き
	"""
	mode = tokenizer.Tokenizer.SplitMode.C
	results = [m.surface() for m in tokenizer_obj.tokenize(mode, text)]

	for mrph in results:
		if not (mrph == ""): #何故か分かち書きの結果として空白データ('')ができたための省く処理
			seikika = tokenizer_obj.tokenize(mode,mrph)[0].normalized_form() #正規化（標準化？）してなるべく言葉の揺れをなくす　e.g. 打込む→打ち込む　かつ丼→カツ丼
			hinsi = tokenizer_obj.tokenize(mode,seikika)[0].part_of_speech()[0]
			if hinsi in ["名詞", "動詞", "助動詞", "形容詞", "指示詞", "感動詞", "接続詞"]: #対象とする品詞を指定
				word = tokenizer_obj.tokenize(mode,seikika)[0].dictionary_form()
				writer.write(word + '\n')


#データを分かち書きしていく
for file in os.listdir('data_perte_20180910'):
	with open('data_perte_20180910/' + file , 'rt') as reader, open('result_data_perte_20180910_hukusi_josi_keidou_nuki/' + file, 'wt') as writer:
		for lines in reader:
			wakati = wakati_by_sudachi(lines,writer) 
