from src.parser import Dataset

def createEmbeddings(dataset : Dataset):
    dataset.setSVDEmbeddings()

if __name__ == "__main__":
    dataset = Dataset(fileName = "model.pkl")
    createEmbeddings(dataset)
