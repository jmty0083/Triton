from numpy import *
import json

def load_stopword(filePath):
    stopwords = []
    with open(filePath,'r', encoding='utf-8') as fr:
        for line in fr.readlines():
          if not len(line) or line.startswith('#'):
              continue
          stopwords.append(line.strip())

    return  stopwords

def load_message(filePath):
    content = []
    label = []
    lines = []

    with open(filePath,'r', encoding='utf-8') as fr:
        for line in fr.readlines():
          if not len(line) or line.startswith('#'):
              continue

          lines.append(line.strip())

        num = len(lines)  # type: int
        for i in range(num):
            message = lines[i].split('\t')
            if(len(message)>=2):
              label.append(message[0])
              content.append(message[1])
    return  num , content, label

def data_storage(content, label,stopword):
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/train_content.json', 'w') as f:
        json.dump(content, f)
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/train_label.json', 'w') as f:
        json.dump(label, f)
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/stopword.json', 'w') as f:
        json.dump(stopword, f)

if '__main__' == __name__:
    num, content, label = load_message('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/train.txt')
    print(num)
    print(len(content))
    stopwords = load_stopword('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/stopword.txt')
    data_storage(content, label,stopwords)