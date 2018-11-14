for d in {1..5};do
    cd data_$d;
    cd train_wakati_toiawase_claim_meisi/claim/
    ls | awk '{nr=NR;}END{for(i=0;i<nr;i++){print("1.0")}}' > ../train_claim_label.txt
    cd ../../;
    
    cd test_wakati_toiawase_claim_meisi/claim/
    ls | awk '{nr=NR;}END{for(i=0;i<nr;i++){print("1.0")}}' > ../test_claim_label.txt
    cd ../../;

    cd ../;
done


    	 

for d in {1..5};do
    cd data_$d;
    cd train_wakati_toiawase_claim_meisi/toiawase/
    ls | awk '{nr=NR;}END{for(i=0;i<nr;i++){print("0.0")}}' > ../train_toiawase_label.txt
    cd ../../;
    
    cd test_wakati_toiawase_claim_meisi/toiawase/
    ls | awk '{nr=NR;}END{for(i=0;i<nr;i++){print("0.0")}}' > ../test_toiawase_label.txt
    cd ../../;

    cd ../;
done


    	 
	
