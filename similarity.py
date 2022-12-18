# Imports
import spacy
nlp = spacy.load("en_core_web_sm")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(round(word1.similarity(word2), 2))
print(round(word3.similarity(word2), 2))
print(round(word3.similarity(word1), 2))


tokens = [token for token in nlp("cat, apple, monkey, banana") if not token.is_punct]

for token1 in tokens:
    for token2 in tokens:
        if token1 != token2:
            print(token1.text, token2.text, round(token1.similarity(token2), 2), sep=" - ")


target_sentence = nlp("Why is my cat on the car?")

sentences = ["Where did my dog go?",
            "Hello, there is my car", 
            "I've lsot my car in my car",
            "I'd like my boat back",
            "I will name my dog Diana"]

for sentence in sentences:
    similarity = nlp(sentence).similarity(target_sentence)
    if similarity != 1:
        print(sentence, round(similarity, 2), sep="\n\t")

"""
Cat and monkey have a higher level of similarity as they are animals
Banana has a higher similarity with monkey than cat because in the 
semantic field of bananas, we are more likely to associate a monkey
with a banana than a cat with a banana. If it were fish instead of banana
I'd expect a similar or higher similarity between cat and fish compared
with monkey and banana
"""

"""
Running with the simpler model yields a warning about the size of the 
model that is being used - it does not include word vectors and thus
it becomes more difficult to accurately compare semantic similarity
Despite this, I didn't notice a change in speed between the two models
"""
