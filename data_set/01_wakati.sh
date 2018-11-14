for f in 1 2 3 4 5 ;do
    cd $f;
    python 01_generate_file.py;
    cd ..;
done
