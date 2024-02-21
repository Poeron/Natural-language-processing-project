from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Cümleleri bir Python listesine yerleştirme
texts = [
    "reported positive result",
    "result covid vaccines",
    "reported close april positive cases deaths",
    "cases reported",
    "many souls already lost lives due way covid",
    "least canadian lives lost due covid misinformation",
    "texans lost lives leadership",
    "already lost faith confidence cdc",
    "million americans died still",
    "million americans died covid",
    "covid million sick million died",
    "meanwhile million americans elsewhere died",
    "find free covid vaccine",
    "find vaccine",
    "havent find covid vaccine today",
    "free vaccine sites",
    "looks pretty ridiculous people dont collapse dead spot covid",
    "collapse pediatric covid surveillance",
    "get accurate test result spot",
    "pretty clear dont think winstonpeters"
]
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)
input_ids = tokenizer(texts, return_tensors="tf", padding=True, truncation=True)["input_ids"]
outputs = model(input_ids, return_dict=True)
for text, prediction in zip(texts, outputs.logits):
    label = tf.argmax(tf.nn.softmax(prediction)).numpy()
    confidence = tf.reduce_max(tf.nn.softmax(prediction)).numpy()
    print(f"Cümle: {text}\nSentiment: {label+1}, Confidence: {confidence}\n")
