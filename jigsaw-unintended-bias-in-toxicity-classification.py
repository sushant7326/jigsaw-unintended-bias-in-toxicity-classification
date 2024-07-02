from google.colab import files
files.upload()

def get_data():
  !pip install -q kaggle
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 /root/.kaggle/kaggle.json
  !kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -p /content
  !unzip \*.zip

get_data()

from google.colab import drive
drive.mount('/content/gdrive/')

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from tqdm import tqdm
from wordcloud import WordCloud


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix,classification_report
# %matplotlib inline
warnings.filterwarnings("ignore")
from IPython.display import Image,YouTubeVideo,HTML

#KERAS Import
from keras.models import Sequential, Model
from keras.utils import to_categorical,plot_model
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers import Embedding, LSTM, GRU, Flatten, Input, concatenate, Conv1D, GlobalMaxPool1D, SpatialDropout1D, GlobalMaxPooling1D, Bidirectional, GlobalAveragePooling1D, add
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Orthogonal
from keras.preprocessing.text import one_hot
from keras.constraints import max_norm
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback

#for attention mechanism
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

df = pd.read_csv('train.csv')
df.head()

test_df = pd.read_csv('test.csv')

print(df.iloc[28]['comment_text'])
print("Toxicity Level: ",df.iloc[28]['target'])

print(df.iloc[4]['comment_text'])
print("Toxicity Level: ",df.iloc[4]['target'])

# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

# Combining all the above statemennts
preprocessed_comments = []
# tqdm is for printing the status bar
for sentence in tqdm(df['comment_text'].values):
    sent = decontracted(sentence)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() )
    preprocessed_comments.append(sent.lower().strip())

if e not in stopwords

df['comment_text'] = preprocessed_comments

df['comment_text'][1]

# Combining all the above statemennts
preprocessed_comments_test = []
# tqdm is for printing the status bar
for sentence in tqdm(test_df['comment_text'].values):
    sent = decontracted(sentence)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split())
    preprocessed_comments_test.append(sent.lower().strip())

test_df['comment_text'] = preprocessed_comments_test

train_len = len(df.index)

miss_val_train_df = df.isnull().sum(axis=0) / train_len
miss_val_train_df = miss_val_train_df[miss_val_train_df > 0] * 100
miss_val_train_df

def plot_features_distribution(features, title):
    plt.figure(figsize=(12,6))
    plt.title(title)
    for feature in features:
        # This will find all the values which are not null so as to plot the ditribution
        sns.distplot(df.loc[~df[feature].isnull(),feature],kde=True,hist=False, bins=120, label=feature)
    plt.xlabel('')
    plt.legend()
    plt.show()

# Distribution of target variable
plt.figure(figsize=(12,6))
plt.title("Distribution of target in the train set")
sns.distplot(df['target'],kde=True,hist=False, bins=120, label='target')
plt.legend()
plt.show()

features = ['severe_toxicity', 'obscene','identity_attack','insult','threat']
plot_features_distribution(features, "Distribution of additional toxicity features in the train set")

features = ['asian', 'black', 'jewish', 'latino', 'other_race_or_ethnicity', 'white']
plot_features_distribution(features, "Distribution of race and ethnicity features values in the train set")

features = ['female', 'male', 'transgender', 'other_gender']
plot_features_distribution(features, "Distribution of gender features values in the train set")

features = ['atheist','buddhist','christian', 'hindu', 'muslim', 'other_religion']
plot_features_distribution(features, "Distribution of religion features values in the train set")

features = ['intellectual_or_learning_disability', 'other_disability', 'physical_disability', 'psychiatric_or_mental_illness']
plot_features_distribution(features, "Distribution of disability features values in the train set")

def make_wordcloud(comment_words,title):
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
              min_font_size = 10,random_state=101,repeat=True).generate(str(comment_words))

    # plot the WordCloud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.title(title)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

make_wordcloud(df.loc[df['insult'] < 0.3]['comment_text'].sample(10000),title='Wordcloud Comments for insult > 0.3')

make_wordcloud(df.loc[df['insult'] > 0.8]['comment_text'].sample(10000),title='Wordcloud Comments for insult > 0.8')

make_wordcloud(df.loc[df['threat'] < 0.3]['comment_text'],title='Wordcloud Comments for threat < 0.3')

make_wordcloud(df.loc[df['threat'] > 0.8]['comment_text'],title='Wordcloud Comments for threat > 0.8')

make_wordcloud(df.loc[df['obscene'] < 0.3]['comment_text'],title='Wordcloud Comments for obscene < 0.3')

make_wordcloud(df.loc[df['obscene'] > 0.8]['comment_text'],title='Wordcloud Comments for obscene > 0.8')

make_wordcloud(df.loc[df['target'] < 0.3]['comment_text'],title='Wordcloud Comments for target < 0.3')

make_wordcloud(df.loc[df['target'] > 0.8]['comment_text'],title='Wordcloud Comments for target > 0.8')

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

for column in identity_columns + ['target']:
    df[column] = np.where(df[column] >= 0.5, True, False)

# Target variable as well
y = df['target'].values

# We will use tfidf and w2v to vectorize the words

train_df, cv_df, y_train, y_cv = train_test_split(df, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2),
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

train_tfidf = vectorizer.fit_transform(train_df["comment_text"])
cv_tfidf = vectorizer.transform(cv_df["comment_text"])
test_tfidf = vectorizer.transform(test_df["comment_text"])

#https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def print_confusion_matrix(confusion_matrix, class_names, figsize = (6,4), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix,index=class_names, columns=class_names
    )
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

# Code help from Slack channel
def threshold_based_prediction(proba,threshold,tpr,fpr):
    thres = threshold[np.argmax(fpr*(1-tpr))]
    predictions = []
    for i in proba:
        if i>=thres:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

alpha = [10 ** x for x in range(-5, 2)]
auc_array_train=[]
auc_array_cv=[]
for i in alpha:
  clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=5,class_weight='balanced')
  clf.fit(train_tfidf, y_train)
  sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
  sig_clf.fit(train_tfidf, y_train)


  predict_y_train = sig_clf.predict_proba(train_tfidf)[:,1]
  predict_y = sig_clf.predict_proba(cv_tfidf)[:,1]
  auc_array_train.append(roc_auc_score(y_train, predict_y_train))
  auc_array_cv.append(roc_auc_score(y_cv, predict_y))
  print('For values of alpha = ', i, "The auc score on CV is:",roc_auc_score(y_cv, predict_y))

# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
fpr_train, tpr_train, threshold_train = roc_curve(y_train, predict_y_train)
fpr_test, tpr_test, threshold_test = roc_curve(y_cv, predict_y)

roc_auc_train = auc(fpr_train, tpr_train)
roc_auc_test = auc(fpr_test, tpr_test)


plt.title('Receiver Operating Characteristic')

plt.plot(fpr_train, tpr_train, 'b', label = 'Training AUC = %0.2f' % roc_auc_train)
plt.plot(fpr_test, tpr_test, 'r', label = 'Test AUC = %0.2f' % roc_auc_test)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

clf = SGDClassifier(alpha=0.00001, penalty='l2', loss='log', random_state=42,class_weight='balanced')
clf.fit(train_tfidf, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(train_tfidf, y_train)

predtrain = sig_clf.predict_proba(train_tfidf)[:,1]
predcv = sig_clf.predict_proba(cv_tfidf)[:,1]
pred = sig_clf.predict_proba(test_tfidf)[:,1]

predtrain = threshold_based_prediction(predtrain,threshold_train,tpr_train,fpr_train)
cm = confusion_matrix(y_train, predtrain)
print("\tTRAIN DATA CONFUSION MATRIX")
print_confusion_matrix(cm,class_names=['NO','YES'])

predcv = threshold_based_prediction(predcv,threshold_test,tpr_test,fpr_test)
cm = confusion_matrix(y_cv, predcv)
print("\tTEST DATA CONFUSION MATRIX")
print_confusion_matrix(cm,class_names=['NO','YES'])

auc = roc_auc_score(y_train, predtrain)
print('\nTRAIN AUC on CV data is %f' % (auc))

auc = roc_auc_score(y_cv, predcv)
print('\nTEST AUC on CV data is %f' % (auc))


print(classification_report(y_cv,predcv))

# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/90986#latest-527331
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive
TOXICITY_COLUMN = 'target'

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

MAX_VOCAB_SIZE = 100000
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'
MAX_SEQUENCE_LENGTH = 300

# Create a text tokenizer.
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_df[TEXT_COLUMN])

# All comments must be truncated or padded to be the same length.
def padding_text(texts, tokenizer):
    return sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

train_text = padding_text(train_df[TEXT_COLUMN], tokenizer)
train_y = to_categorical(train_df[TOXICITY_COLUMN])
validate_text = padding_text(cv_df[TEXT_COLUMN], tokenizer)
validate_y = to_categorical(cv_df[TOXICITY_COLUMN])

# for submission purpose
test_text = padding_text(test_df[TEXT_COLUMN], tokenizer)

NUM_EPOCHS = 10
BATCH_SIZE = 512

# https://fasttext.cc/docs/en/english-vectors.html
!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

!unzip crawl*.zip

embeddings_index = {}
with open('crawl-300d-2M.vec' ,encoding='utf8') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

len(tokenizer.word_index)

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,300))
num_words_in_embedding = 0
for word, i in tokenizer.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    num_words_in_embedding += 1
    # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector

embedding_matrix.shape

input_text = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
x = embedding_layer(input_text)
x = Conv1D(128, 2, activation='relu', padding='same')(x)
x = MaxPooling1D(5, padding='same')(x)
x = Conv1D(128, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(5, padding='same')(x)
x = Conv1D(128, 4, activation='relu', padding='same')(x)
x = MaxPooling1D(40, padding='same')(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=[input_text], outputs=[output])

plot_model(model, show_shapes=True, to_file='model.png')

Image(filename="model.png")

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

CNN_Model = model.fit(train_text,train_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_y))

# Prediction on CV data
MODEL_NAME = 'cnn_model'
cv_df[MODEL_NAME] = model.predict(validate_text)[:, 1]

cv_df.head(3)

bias_metrics_df = compute_bias_metrics_for_model(cv_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df

get_final_metric(bias_metrics_df, calculate_overall_auc(cv_df, MODEL_NAME))


input_text_lstm = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer_lstm = Embedding(len(tokenizer.word_index) + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
l = embedding_layer_lstm(input_text_lstm)
l = LSTM(128,return_sequences=True,dropout=0.5,kernel_regularizer=l2(0.001))(l)
l = Flatten()(l)
l = Dropout(0.5)(l)
l = Dense(128, activation='relu')(l)
lstm_output = Dense(2, activation='softmax')(l)

model = Model(inputs=[input_text_lstm], outputs=[lstm_output])


plot_model(model, show_shapes=True, to_file='singlelstm.png')
Image(filename="singlelstm.png")

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

SLSTM_Model = model.fit(train_text,train_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_y))


# Prediction on CV data
MODEL_NAME = 'slstm_model'
cv_df[MODEL_NAME] = model.predict(validate_text)[:, 1]

bias_metrics_df = compute_bias_metrics_for_model(cv_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df

get_final_metric(bias_metrics_df, calculate_overall_auc(cv_df, MODEL_NAME))

#Refer : https://datascience.stackexchange.com/questions/31129/sample-importance-training-weights-in-keras

input_text_blstm = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer_blstm = Embedding(len(tokenizer.word_index) + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
b = embedding_layer_blstm(input_text_blstm)
b = SpatialDropout1D(0.2)(b)
b = Bidirectional(LSTM(128, return_sequences=True))(b)
b = Bidirectional(LSTM(128, return_sequences=True))(b)
b = GlobalMaxPooling1D()(b)
b = Dense(512, activation='relu')(b)
b = Dense(512, activation='relu')(b)
blstm_output = Dense(2, activation='softmax')(b)

model = Model(inputs=[input_text_blstm], outputs=[blstm_output])


plot_model(model, show_shapes=True, to_file='blstm.png')
Image(filename="blstm.png")


# https://www.kaggle.com/kernels/scriptcontent/16109977/download
sample_weights = np.ones(len(train_text), dtype=np.float32)
sample_weights += train_df[identity_columns].sum(axis=1) * 3
sample_weights += train_df[TOXICITY_COLUMN] * (~train_df[identity_columns]).sum(axis=1) * 3
sample_weights += (~train_df[TOXICITY_COLUMN]) * train_df[identity_columns].sum(axis=1) * 9
sample_weights /= sample_weights.mean()

[sample_weights.values, np.ones_like(sample_weights)]

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

NUM_EPOCHS = 1

BLSTM_Model = model.fit(train_text,train_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_y),
              sample_weight = [sample_weights.values])

# Prediction on CV data
MODEL_NAME = 'blstm_model'
cv_df[MODEL_NAME] = model.predict(validate_text)[:, 1]

bias_metrics_df = compute_bias_metrics_for_model(cv_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df

get_final_metric(bias_metrics_df, calculate_overall_auc(cv_df, MODEL_NAME))


WBLSTM_Model = model.fit(train_text,train_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_y),
              sample_weight = [sample_weights.values])


# Prediction on CV data
MODEL_NAME = 'blstm_model_weighted'
cv_df[MODEL_NAME] = model.predict(validate_text)[:, 1]

bias_metrics_df = compute_bias_metrics_for_model(cv_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df

get_final_metric(bias_metrics_df, calculate_overall_auc(cv_df, MODEL_NAME))

input_text_bgru = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer_bgru = Embedding(len(tokenizer.word_index) + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
g = embedding_layer_bgru(input_text_bgru)
g = SpatialDropout1D(0.4)(g)
g = Bidirectional(GRU(64, return_sequences=True))(g)
att = Attention(MAX_SEQUENCE_LENGTH)(g)
g = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(g)
avg_pool1 = GlobalAveragePooling1D()(g)
max_pool1 = GlobalMaxPooling1D()(g)
g = concatenate([att,avg_pool1, max_pool1])
g = Dense(128, activation='relu')(g)
bgru_output = Dense(2, activation='softmax')(g)

model = Model(inputs=[input_text_bgru], outputs=[bgru_output])

plot_model(model, show_shapes=True, to_file='bgru.png')
Image(filename="bgru.png")

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

NUM_EPOCHS = 6
BATCH_SIZE = 1300

BGRU_Model = model.fit(train_text,train_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_y))


temp_df = cv_df[:30000]

# Prediction on CV data
MODEL_NAME = 'bgru_model'
cv_df[MODEL_NAME] = model.predict(validate_text)[:, 1]

bias_metrics_df = compute_bias_metrics_for_model(cv_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df

get_final_metric(bias_metrics_df, calculate_overall_auc(cv_df, MODEL_NAME))

input_text_bgru = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32')
embedding_layer_bgru = Embedding(len(tokenizer.word_index) + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
g = embedding_layer_bgru(input_text_bgru)
g = SpatialDropout1D(0.4)(g)
g = Bidirectional(GRU(128, return_sequences=True))(g)
g = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(g)
avg_pool1 = GlobalAveragePooling1D()(g)
max_pool1 = GlobalMaxPooling1D()(g)
g = concatenate([avg_pool1, max_pool1])
#g = Dense(128, activation='relu')(g)
bgru_output = Dense(2, activation='softmax')(g)

model = Model(inputs=[input_text_bgru], outputs=[bgru_output])

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

NUM_EPOCHS = 5
BATCH_SIZE = 1024

BGRU_Model = model.fit(train_text,train_y,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_y))

# Prediction on CV data
MODEL_NAME = 'bgru_model'
cv_df[MODEL_NAME] = model.predict(validate_text)[:, 1]

bias_metrics_df = compute_bias_metrics_for_model(cv_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
bias_metrics_df

get_final_metric(bias_metrics_df, calculate_overall_auc(cv_df, MODEL_NAME))

"""### Conclusion
1. GRU + CONV1D giving us the best score of 0.92311
"""

from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Model", "AUC/Designed Metrics AUC"]

x.add_row(["Logistic Regression:", 0.862040])
x.add_row(["CNN1D:", 0.89921])
x.add_row(["Single LSTM:", 0.89822])
x.add_row(["Bi-directional LSTM:", 0.88767])
x.add_row(["Bi-directional LSTM with sample weights:", 0.88839])
x.add_row(["Bi-directional GRU + CONV1D:", 0.92311])
x.add_row(["Bi-directional GRU + CONV1D MORE DENSE:", 0.91713])

print(x)