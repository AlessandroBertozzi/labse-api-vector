import torch
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F

def similarity(embeddings_1, embeddings_2):
    normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
    normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
    return torch.matmul(
        normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )

class LaBSE:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = BertModel.from_pretrained("setu4993/LaBSE")
        self.model.eval()

    @torch.no_grad()
    def __call__(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]
        tokens = self.tokenizer(sentences, return_tensors="pt", padding=True)
        outputs = self.model(**tokens)
        embeddings = outputs.pooler_output
        return F.normalize(embeddings, p=2).cpu().numpy()

    @property
    def dim(self):
        return 768

if __name__ == "__main__":
    labse = LaBSE()
    print(labse(["odi et amo", "quare id faciam"]).shape)