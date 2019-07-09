#! /bin/bash
# a test script

score answer.csv -m ref.py -c -s -d -abcd -g gakurui_id.py -n &> stat-mksheet.txt
score answer.csv -m ref.py > mksheet.csv
score mksheet.csv -j sup.csv -d -n
score mksheet.csv -j sup.csv -d -adjust 50 70 90 -n
score mksheet.csv -j sup.csv -d -adjust 50 70 90 > mksheet-sup-adjust.csv
score SIToroku.csv -t -j mksheet-sup-adjust.csv -d -i 0 100 -o twins-upload.csv
score twins-upload.csv -t -d -abcd -g gakurui_id.py -s -n &> stat-twins.txt

