
from io import open
from conllu import parse_incr

fn = "/Users/ar/work/ud-portuguese-bosque/documents/CF0001.conllu"
data_file = open(fn, "r", encoding="utf-8")
for tokenlist in parse_incr(data_file):
    print(dict(tokenlist[0]))
    
    
    
