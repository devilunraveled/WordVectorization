from src.parser import Dataset
from src.parser import Word2Vec

def createEmbeddings(dataset : Dataset):
    dataset.setWord2VecEmbeddings()

if __name__ == "__main__":
    dataset = Dataset(fileName = "model.pkl")
    createEmbeddings(dataset)
