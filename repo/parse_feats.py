import stanza
import pandas as pd
import numpy as np
# stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

data = pd.read_csv("/home/mahasweta/Desktop/igextract/data/policy.csv")

statements = data["policy.statement"].tolist()
doc_class = data["policy.doc.name"].tolist()
section = data["section.name"].tolist()
class_ = data["ostrom.statement.class"].tolist()
category = data["rule.category"].tolist()
sub_category = data["rule.subcategory"].tolist()

features = pd.DataFrame(columns=["id","sid","tid","word","pos","relation","word_source", \
            "document","section","class","category","subcategory","CodeType"])

for line,document,sec,cls_,cat,subcat in zip(statements,doc_class,section,class_,category,sub_category):
    doc = nlp(line.lower())

    # for sentence in doc.sentences:
    tokens = ['ROOT'] + [word.text for sentence in doc.sentences for word in sentence.words ]
    words = ['ROOT'] + [word for sentence in doc.sentences for word in sentence.words]
    for id_,word in enumerate(words):
        if word == 'ROOT':
            features.loc[len(features)] = ["doc1",1,id_,"ROOT","","NA","NA","NA","NA","NA","NA","NA",1]
        else:
            features.loc[len(features)] = ["doc1",1,id_,word.text,word.xpos,word.deprel,tokens[word.head],document,sec,cls_,cat,subcat,1]

print(features.head(10))
print(features.shape[0] - features.dropna().shape[0])
features.to_csv("/home/mahasweta/Desktop/igextract/data/asf_with_code.csv",index = False)