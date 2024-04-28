from src.classification import RNNClassfierDatset, RNNClassifier, prepareDataPointsFromData
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
    dataset = Dataset(trainPath="./corpus/sample.csv")
    # dataset = Dataset(fileName = "test.pkl")
    startTime = time.time()
    print(f"Time taken to train word2vec : {time.time() - startTime:.2f}s")
    print(f"back embedding : {dataset.word2Vec.getWordEmbedding(word = 'his')}")

def fullTest():
    dataset = Dataset(fileName = "model.pkl")
    startTime = time.time()
    dataset.performSVD()
    print(f"Time taken to perform SVD : {time.time() - startTime:.2f}s")
    dataset.prepareWord2VecEmbedding()
    print(f"Time taken to prepare word2vec embedding : {time.time() - startTime:.2f}s")

def embeddingTest():
    dataset = Dataset(fileName = "model.pkl")
    svdEm = dataset.getSVDEmbedding('his')
    w2vEm = dataset.getWord2VecEmbedding('his')
    print("SVD : ", svdEm[0:10])
    print("w2v : ", w2vEm[0:10])


def classification():
    dataset = Dataset(fileName = "model.pkl")
    ## prepare data 
    dataset.getData()
    dataset.getTestData()
    data = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.labels, dataset.tokenizedData)]
    testData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.testLabels, dataset.testData)]
    trainData = prepareDataPointsFromData(data = data, embeddingLookUp = dataset.getWord2VecEmbedding)
    classifier = RNNClassifier(inputSize=256, embeddingMap=dataset.getWord2VecEmbedding, data=RNNClassfierDatset(trainData), testData=RNNClassfierDatset(testData))
    classifier.createDataset()
    classifier.trainModel()
    acc = classifier.evaluate()
    print("Classification Accuracy : ", acc)

if __name__ == "__main__":
    # vocabTest()
    # contextTest()
    # coOcMatrixSparse()
    # SparseSVDTest()
    # SVDEmbeddingTest()
    # trainWord2VecTest()
    # fullTest()
    # embeddingTest()
    classification()
