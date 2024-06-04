rm -r CHASEDB1
rm -r CHASEDB1_GT
rm -r dataset
unzip -d CHASEDB1 CHASEDB1.zip
python3 my_dataset.py 
python3 main.py --model_type=R2U_Net