def model_bert():

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer, make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
    from sklearn import set_config; set_config(display='diagram')
    from tensorflow.keras.preprocessing.text import Tokenizer
    import string
    import os
    import nltk
    nltk.download('stopwords')
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import layers, Sequential
    from transformers import AutoTokenizer
    from tensorflow.keras.callbacks import EarlyStopping
    import datasets
    import transformers
    data = pd.read_csv("/Users/lucaspicot/code/Zen1400/fake_news/raw_data/cleaned_df.csv")
    ## Delete the first column
    data = data.drop(columns= 'Unnamed: 0')
    data = data.dropna()

    ##Define the train test split

    train_val_df = data.sample(frac = 0.8)
    test_df = data.drop(train_val_df.index)

    train_df = train_val_df.sample(frac = 0.8)
    val_df = train_val_df.drop(train_df.index)

    df_train = datasets.Dataset.from_pandas(train_df)
    df_test = datasets.Dataset.from_pandas(test_df)
    my_dataset_dict = datasets.DatasetDict({"train":df_train,"test":df_test})

    ## tokenization

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_data = my_dataset_dict.map(preprocess_function, batched = True)

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    ##train set and validation set

    tf_train_set = tokenized_data["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = tokenized_data["test"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "label"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    #parameters

    from transformers import create_optimizer
    import tensorflow as tf

    batch_size = 16
    num_epochs = 5

    es = EarlyStopping(patience = 5, verbose=2, monitor='val_loss', restore_best_weights = True)

    batches_per_epoch = len(tokenized_data["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    #import pre-trained model bert base uncased

    from transformers import TFAutoModelForSequenceClassification


    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    #compile the model

    import tensorflow as tf

    model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics ='accuracy')

    model.bert.trainable = False

    #train the model

    model.fit(x=tf_train_set, validation_data=tf_validation_set, callbacks = [es], epochs=30, batch_size=batch_size)

    model.save("/home/lucaspicot/fake_news/model_bert")




if __name__ == '__main__':
    model_bert()
