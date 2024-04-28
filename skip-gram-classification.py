# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
"""
Classification Testing.
"""

# %% 
from src.classification import RNNClassifier
from src.parser import Dataset
from src.Config import Constants, Word2VecConfig

# %%
dataset = Dataset(fileName = "model.pkl", word2VecLoad = True)

# %%
dataset.getData()

# %%
# Only set this if word2Vec embeddings are not found.
# dataset.setWord2VecEmbeddings()


# %%
dataset.getTestData()

# %%
trainData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.labels, dataset.tokenizedData)]
testData = [ (label, tuple(tokenizedSentence)) for label, tokenizedSentence in zip(dataset.testLabels, dataset.testData)]

# %%
# trainData[0][1][1]

# %%
# print(dataset.getWord2VecEmbedding(trainData[0][1][1]))

# %% [markdown]
"""
### Using Word2Vec Embeddings for Classification
"""

# %%
classifier = RNNClassifier(inputSize=256, embeddingMap=dataset.getWord2VecEmbedding, 
                           data=trainData, testData=testData, 
                           fileName = f"w2vClassifier_{Word2VecConfig.contextWindow}.pt", learningRate=8e-5)

# %%
classifier.createDataset()

# %%
print(Word2VecConfig.contextWindow)

# %% 
# classifier.trainModel()

# %%
classifier.loadModel(fileName = classifier.fileName)

# %%
print(f"Best Performing Model with Accuracy : {classifier.evaluation()}")

# %%
classifier.plotConfusionMatrix(name=f"W2V_{Constants.EmbeddingSize}_{Word2VecConfig.contextWindow}")
