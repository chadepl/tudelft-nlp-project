import json
import nltk
import numpy as np

# FUNCTIONS (Separate in files)

# Data loading

""" Fields in instances.jsonl:
 { 
    "id": "<instance id>",
    "postTimestamp": "<weekday> <month> <day> <hour>:<minute>:<second> <time_offset> <year>",
    "postText": ["<text of the post with links removed>"],
    "postMedia": ["<path to a file in the media archive>"],
    "targetTitle": "<title of target article>",
    "targetDescription": "<description tag of target article>",
    "targetKeywords": "<keywords tag of target article>",
    "targetParagraphs": ["<text of the ith paragraph in the target article>"],
    "targetCaptions": ["<caption of the ith image in the target article>"]
  } """

""" Fields in truth.jsonl:
  {
    "id": "<instance id>",
    "truthJudgments": [<number in [0,1]>],
    "truthMean": <number in [0,1]>,
    "truthMedian": <number in [0,1]>,
    "truthMode": <number in [0,1]>,
    "truthClass": "clickbait | no-clickbait"
  } """

def loadDataset(size):

    instances = []
    labels = []

    fileName = 'trainSmall' if size == 'small' else 'trainLarge'

    with open('data/'+fileName+'/instances.jsonl') as file:
        for line in file:
            instances.append(json.loads(line))

    with open('data/'+fileName+'/truth.jsonl') as file:
        for line in file:
            labels.append(json.loads(line))

    return instances, labels

#def getNDArray()

# Feature generation (assume inputs and outputs are nd )

def addFeature(inputArray, feature):    
    rows, cols = inputArray.shape
    a = np.random.rand(rows,cols)
    b = np.zeros((rows,cols+1))
    b[:,:-1] = a
    b[:, cols-1] = feature
    return b

def numChars(instances, colName):
    result = []
    for i in instances:
        result.append(len(i[colName]))
    return result

def numWords(instances, colName):
    result = []
    for i in instances:        
        result.append(len(nltk.word_tokenize(i[colName])))
    return result

def detectUpperCaseWords(instances, colName):
    result = []
    for i in instances:
        if len([word for word in i[colName] if word.isupper()]) > 0:
            result.append(1)
        else:
            result.append(0)        
    return result


# MAIN CODE

instances, labels = loadDataset('small')

print('Number of instances loaded: ' + str(len(instances)))
# - print('Example instance: ' + str(instances[0]))
print('Number of labels loaded: ' + str(len(labels)))
# - print('Example labels: ' + str(labels[0]))


train = np.zeros((len(instances), 1), dtype=float)
train[:, 0] = [float(x['truthMean']) for x in labels] # choose the mean
train = addFeature(train, detectUpperCaseWords(instances, 'postText')) # UPPERCASE in post text
#train = addFeature(train, numWords(instances, 'postText')) # numWords of post text

print(train[:,1])