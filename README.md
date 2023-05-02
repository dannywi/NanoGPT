
# Nano GPT (WIP)
Following Karpathy's tutorial https://www.youtube.com/watch?v=kCc8FmEb1nY

## Setups
```
# todo: get a proper input file. samples:
# curl -o input.txt https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
# curl -o input.txt https://norvig.com/big.txt

# python version 3.10.6
# create venv, get in, and install libraries ('deactivate' to get out)
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements
```

## Running
```bash
python bigram.py
```

# Latest results
```bash
step 0: train loss 4.6919, val loss 4.7017
step 500: train loss 2.2137, val loss 2.4000
step 1000: train loss 1.9264, val loss 2.1980
step 1500: train loss 1.7712, val loss 2.0667
step 2000: train loss 1.6794, val loss 1.9822
step 2500: train loss 1.6112, val loss 1.8829
step 3000: train loss 1.5645, val loss 1.8539
step 3500: train loss 1.5334, val loss 1.8537
step 4000: train loss 1.4996, val loss 1.8321
step 4500: train loss 1.4804, val loss 1.7787
FINAL: train loss 1.4804, val loss 1.7787
==== GENERATED TEXT ====
        power Like heldly aside-re for shovere to
chinn the jout brearly themselvant greeath; etach.
He too, fear the softer
vitalent with other regulating serily possible stops, or placking had
reable, without heard to his eyes. How in attated and place, and where
Rostovs, I will His having it organ
from the parch. Norma," Rostow
and how and helre would Nulatashed taken the prividents of swring and coverder but, dening and
inducts appears and, happen bruine to the aborman
becomang the even term diseases boncommocs.

Learly which is as the calm of an everything, and
the run of want-king that stepping to considers.

#trad Territ It myclino speverings.
For which which smywhered their bath but the
ay pontox in a ster of Innuman's blandy, the prespockilish nothern met frosury
it, and a fixeduction, early fingerlyzed of the lad steber of the
shouts of cattor.
Unnoxin. In the rarge escrumbts and espect of their wading be
side indected liables of Mostopp. That in greamen a full. Foldth doescrived how

real    33m50.159s
user    94m49.242s
sys     18m31.607s
```

## Todo
* use formatter
