import tensorflow_datasets as tfds
import tensorflow as tf
import os

BUFFER_SIZE = int(os.environ["BUFFER_SIZE"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])


class Pipeline(object):
    def __init__(self, src_txt, encoder_path, encoder=None, shuffle=True, max_num_words=200):
        self.encoder = encoder 
        self.src_txt = src_txt
        self.encoder_path = encoder_path
        self.tokenizer = tfds.features.text.Tokenizer(reserved_tokens=["<B>"])
        self.shuffle = shuffle
        self.ds = None
        self.vocab_set = None
        self.max_num_words = max_num_words
    
    def label(self, max_num_words):
        """

        :param max_num_words: if the sequence is longer, it will be truncated
        :return:
        """
        with open(self.src_txt, "r") as f:
            for i,l in enumerate(f):
                l = l.rstrip()
                tokenized = self.tokenizer.tokenize(l)
                if len(tokenized) > max_num_words:
                    tokenized = tokenized[:max_num_words]
                l = " ".join(tokenized)
                if len(l.replace(' ', '')):
                    if not l.endswith('<B>'):
                        # this hack: deals with the situation when a line does
                        # not end with a boundary token
                        l = l + " <nobound>"
                    split = l.split(" ")
                    l_labels = []
                    for idx, t in enumerate(split):
                        if t == "<B>":
                            continue
                        else:
                            try:
                                if split[idx + 1] == "<B>":
                                    l_labels.append(1)
                                else:
                                    l_labels.append(0)
                            except IndexError:
                                continue
                    l = ' '.join(l.replace('<B>', '').split())
                    if not len(l):
                        continue
                    if l.endswith(" <nobound>"):
                        l = l.replace(" <nobound>", "")
                    yield l, l_labels
    
    def get_vocab(self):
        vocab_set = set()
        with open(self.src_txt, "r") as f:
            line = f.readline()
            while line:
                line = line.rstrip()
                line = ' '.join(line.replace('<B>', '').split())
                some_tokens = self.tokenizer.tokenize(line)
                vocab_set.update(some_tokens)
                line = f.readline()
        self.vocab_set = vocab_set
        return vocab_set
    
    
    def get_encoder(self):
        if os.path.isfile(self.encoder_path):
            return tfds.features.text.TokenTextEncoder.load_from_file(self.encoder_path)          
        encoder = tfds.features.text.TokenTextEncoder(self.vocab_set)
        self.encoder = encoder
        return encoder
    
    
    def save_encoder(self):
        self.encoder.save_to_file(self.encoder_path)
        return
    
    
    def encode(self, text_tensor, labels):
        encoded_text = self.encoder.encode(text_tensor.numpy())
        return encoded_text, labels
    
    
    def encode_map_fn(self, text, labels):
        return tf.py_function(self.encode, inp=[text,labels], Tout=(tf.int64,tf.int32))
  
    def get_max_seq_len_and_num_elems(self):
        """
        WARNING: very inefficient
        """
        ds = tf.data.Dataset.from_generator(self.label, args=[self.max_num_words],
                                       output_types=(tf.string, tf.int32),
                                       output_shapes=((None), (None)), )
        num_elems = 0
        max_seq_len = 0
        for elem in ds:
            num_elems += 1
            this_seq_len = len(elem[1].numpy())
            max_seq_len = this_seq_len if this_seq_len > max_seq_len else max_seq_len
        return max_seq_len, num_elems
    
    def get_dataset(self, batch=True, padded_shapes=([None],[None])):
        ds = tf.data.Dataset.from_generator(self.label, args=[self.max_num_words],
                                       output_types=(tf.string, tf.int32),
                                       output_shapes=((None), (None)), )
        self.get_vocab()
        if self.encoder is None:
            self.get_encoder()
            self.save_encoder()
        ds = ds.map(self.encode_map_fn)
        if self.shuffle:
            ds = ds.shuffle(BUFFER_SIZE) 
        if batch:
            ds = ds.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)
        self.ds = ds
        return ds, self.encoder
