#! /bin/bash
# a test script

score data/answer.csv -m data/ref.py 80 -c -s -d -abcd -g data/meibo.csv -n &> stat-mksheet.txt
score data/answer.csv -m data/ref.py 80 > mksheet.csv
score mksheet.csv -j data/sup.csv > mksheet-sup.csv
score mksheet-sup.csv -d -adjust 50 70 90 -n
score mksheet-sup.csv -adjust 50 70 90 -i 0 100 > mksheet-sup-adjust.csv
score data/SIToroku.csv -t -j mksheet-sup-adjust.csv -d -o twins-upload.csv
score twins-upload.csv -t -d -abcd -g data/meibo.csv -s -n &> stat-twins.txt
score data/meibo.csv -r mksheet.csv data/sup.csv mksheet-sup.csv mksheet-sup-adjust.csv -o record.xlsx

