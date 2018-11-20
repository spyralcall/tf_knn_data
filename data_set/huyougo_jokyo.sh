#不要語の除去・抽出を行うシェルスクリプト
#まずは不要語の除去

#全データの学習データにおいて
for d in {1..5};do 
    cd data_$d;
    cd train_wakati_toiawase_claim_meisi;
    mkdir claim_jokyo;
    mkdir toiawase_jokyo;
    cd claim;
    
    #クレームの不要語除去
    for f in * ;do
        grep  -v [0-9] $f |\
	    grep -v [０-９]|\
	    grep -v [a-z] |\
	    grep -v [A-Z] |\
	    grep -v [ａ-ｚ]|\
	    grep -v [Ａ-Ｚ]|\
	    #半角カナを全角カナに変換
	    nkf -w > ../claim_jokyo/jokyo_$f
    done

    #クレームの不要語の書き出し
    for f in *;do
	grep -e [0-9] -e [０-９] -e [a-z] -e [A-Z] -e [ａ-ｚ] -e [Ａ-Ｚ] $f
    done > ../huyougo.csv
        
    cd ../toiawase;
    
    #問い合わせの不要語除去
    for f in * ;do
        grep  -v [0-9] $f |\
	    grep -v [０-９]|\
	    grep -v [a-z] |\
	    grep -v [A-Z] |\
	    grep -v [ａ-ｚ]|\
	    grep -v [Ａ-Ｚ]|\
	    #半角カナを全角カナに変換
	    nkf -w > ../toiawase_jokyo/jokyo_$f
    done
    
    #問い合わせの不要語の書き出し
    for f in *;do
	grep -e [0-9] -e [０-９] -e [a-z] -e [A-Z] -e [ａ-ｚ] -e [Ａ-Ｚ] $f
    done >> ../huyougo.csv

    cd ../../../;
done





#######################################################################################
#全データのテストデータにおいて
for d in {1..5};do
    cd data_$d;
    cd test_wakati_toiawase_claim_meisi;
    mkdir claim_jokyo;
    mkdir toiawase_jokyo;
    
    cd claim;
    
    #クレームの不要語除去
    for f in * ;do
        grep  -v [0-9] $f |\
	    grep -v [０-９]|\
	    grep -v [a-z] |\
	    grep -v [A-Z] |\
	    grep -v [ａ-ｚ]|\
	    grep -v [Ａ-Ｚ]|\
	    #半角カナを全角カナに変換
	    nkf -w > ../claim_jokyo/jokyo_$f
    done
    
    #クレームの不要語の書き出し
    for f in *;do
	grep -e [0-9] -e [０-９] -e [a-z] -e [A-Z] -e [ａ-ｚ] -e [Ａ-Ｚ] $f
    done > ../huyougo.csv
    
    cd ../toiawase;
    
    #問い合わせの不要語除去
    for f in * ;do
        grep  -v [0-9] $f |\
	    grep -v [０-９]|\
	    grep -v [a-z] |\
	    grep -v [A-Z] |\
	    grep -v [ａ-ｚ]|\
	    grep -v [Ａ-Ｚ]|\
	    #半角カナを全角カナに変換
	    nkf -w > ../toiawase_jokyo/jokyo_$f
    done
    
    #問い合わせの不要語の書き出し
    for f in *;do
	grep -e [0-9] -e [０-９] -e [a-z] -e [A-Z] -e [ａ-ｚ] -e [Ａ-Ｚ] $f
    done >> ../huyougo.csv

    cd ../../../;
done

