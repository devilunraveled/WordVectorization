from pathlib import Path
import os

class Structure:
    """
        Information about the project structure, file management.
    """
    # Corpus is in the parent directory.
    corpusPath = os.path.join(Path(__file__).absolute().parents[1], "corpus/")
    
    # Train Path : corpus/train.csv
    trainPath = str(os.path.join(corpusPath, "train.csv"))

    # Test Path : corpus/test.csv
    testPath = str(os.path.join(corpusPath, "test.csv"))

class Constants:
    # Custom tokens
    startToken = "<s>"
    endToken = "</s>"
    unkToken = "<unk>"
    customTokens = {startToken, endToken, unkToken}

    cleanser = r'(\\+|\/\/+)'
    
class SVDConfig(Constants):
    # NUmber of words to use for stochastic SVD.
    numWords = 1000
    
    # Embeddding Size
    EmbeddingSize = 100

    # Context window
    contextWindow = 2


class Word2VecConfig(Constants):
    # Embeddding Size
    EmbeddingSize = 100

    # Negative Samples
    negativeSamples = 1

    #number of epochs
    epochs = 10

    #batch size
    batchSize = 65568
