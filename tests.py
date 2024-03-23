from src.parser import Dataset
import time


def vocabTest():
    dataset = Dataset()
    startTime = time.time() 
    vocab = dataset.getVocabulary()
    print(f"Time taken to get the vocab : {time.time() - startTime:.2f}s")
    print(f"Size of vocab : {len(vocab)}")

def contextTest():
    # dataset = Dataset(trainPath="./corpus/sample.csv")
    dataset = Dataset()
    startTime = time.time()
    dataset.__getWordContext__(word = "not")
    print(f"Time taken to get the context : {time.time() - startTime:.2f}s")

def coOcMatrixSparse():
    # dataset = Dataset(trainPath="./corpus/sample.csv")
    dataset = Dataset()
    dataset.getData()
    startTime = time.time()
    coOcc = dataset.getCoocurenceMatrix(sparse = True)
    print(f"Time taken to get the sparse co-oc : {time.time() - startTime:.2f}s")

def SparseSVDTest():
    # dataset = Dataset(trainPath="./corpus/sample.csv")
    dataset = Dataset()
    startTime = time.time()
    dataset.performSVD(sparse = True)
    print(f"Time taken to perform SVD : {time.time() - startTime:.2f}s")
    
    # print(f"U : {dataset.U}\nS : {dataset.S}\nVt : {dataset.Vt}")

def SVDEmbeddingTest():
    # dataset = Dataset(trainPath="./corpus/sample.csv")
    dataset = Dataset()
    startTime = time.time()
    dataset.performSVD()
    print(f"Time taken to perform SVD : {time.time() - startTime:.2f}s")
    wordEmbedding = dataset.getWordEmbedding(word = "his")
    print(wordEmbedding)
    print(len(wordEmbedding))
       
def trainWord2VecTest():
    # dataset = Dataset(trainPath="./corpus/sample.csv")
    dataset = Dataset(fileName = "test.pkl")
    startTime = time.time()
    dataset.getWord2VecEmbedding()
    print(f"Time taken to train word2vec : {time.time() - startTime:.2f}s")

if __name__ == "__main__":
    trainWord2VecTest()
