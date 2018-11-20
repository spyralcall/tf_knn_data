for f in 1 2 3 4 5 ;do
    cd data_$f;
    python generate_wakati_file.py;
    cd ..;
done
