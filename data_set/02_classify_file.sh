for f in {1..5};do
    cd data_$f;
    cd train_wakati_toiawase_claim_meisi;
    mkdir claim;
    mkdir toiawase;
    for l in *_A0*.txt;do
	mv $l claim/;
    done

    for s in *_C0*.txt;do
	mv $s toiawase/;
    done
    cd ../../;
done


for f in {1..5};do
    cd data_$f;
    cd test_wakati_toiawase_claim_meisi;
    mkdir claim;
    mkdir toiawase;
    for l in *_A0*.txt;do
	mv $l claim/;
    done

    for s in *_C0*.txt;do
	mv $s toiawase/;
    done
    cd ../../;
done
