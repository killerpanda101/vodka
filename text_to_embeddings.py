import torch
import numpy as np
from numpy.linalg import norm
import torch
from angle_emb import AnglE

class TextToEmbeddings:

    def __init__(self, model='./models/UAE'):
        self.angle = AnglE.from_pretrained(model_name_or_path=model, pooling_strategy='cls')
        if torch.cuda.is_available():
            self.angle = AnglE.from_pretrained(model_name_or_path=model, pooling_strategy='cls').cuda()
        elif torch.backends.mps.is_available():
            self.angle = AnglE.from_pretrained(model_name_or_path=model, pooling_strategy='cls').to('mps')


    # not using prompt for retrieval 
    def encode(self, text):
        return self.angle.encode(text, to_numpy=True)
    
    
    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)) 
    

if __name__ == "__main__":
    embed_model = TextToEmbeddings()
    prompt = "This picture describes a telephone booth in a park with a blue sky."
    item1 = "This is a red telephone booth on a black background."
    item2 = "This is the picture of a blue sky with white cloud's."
    item3 = "The scene describes a cobble stone wall against a black background."
    item4 = "The is a picture of a red sports car against a bridge"
    item5 = "the picture is of a man in a background full of stars."

    texts = [item1, item2, item3, item4, item5]
    prompt_vec = embed_model.encode(prompt)
    item_vecs = embed_model.encode(texts)

    for idx, item_vec in enumerate(item_vecs):
        print(f'{idx+1} item and prompt similarity: {embed_model.cosine_similarity(prompt_vec, item_vec)}')