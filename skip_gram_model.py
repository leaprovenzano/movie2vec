import math
import tensorflow as tf

class SkipGramModel(object):

    """Word2Vec ( Mikolov et al.) using skipgram model. Can be used on non-word text
    inputs too!
    
    Attributes:
        b (tf.Variable): bias for nce loss
        batch_size (int): size of training batch
        embedding_dims (int): size of embedding
        embeddings (tf.Variable): embedding object for text/items
        init_op (tf.global_variables_initializer): this is tf.global_variables_initializer 
            to initilize variable in the model. call as model.init_op() in session.
        loss (TYPE): model loss function - mean NCE loss
        lr (TYPE): learning rate passed to optimizer 
        n_samples (int): number of negative samples based on batch size and
            the sample_factor parameter.
        normalized_embeddings (tensor): normalized embeddings
        optimize (tensor): optimization step
        optimizer (tf.train.Optimizer): optimizer default is adagrad
        saver (tf.train.Saver): tf.train.Saver object call in session after model.init_op().
        similarity (tensor): measure similarity on val_x
        val_data (np.array, 'int32'): indexes of validation samples to evaluate similarity on during training
        val_x (tf.Placeholder): from val_data
        vocab_size (int): size of 'vocabulary'
        w (tf.Variable): weights for nce loss
        x (tf.Placeholder): placeholder for input data
        y (tf.Placeholder): placeholder for input labels
    """
    
    def __init__(self, vocab_size, val_data, embedding_dims=256, batch_size=128, sample_factor=1, lr=1., optimizer=tf.train.AdagradOptimizer):
        
        self.embedding_dims = embedding_dims
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.val_data = val_data

        
        #number of negative samples (batch * sample factors)
        self.n_samples = int(round(sample_factor  * self.batch_size))
        self.lr = lr 
        self.init_model()
        
    @staticmethod
    def embed(embeddings, inp):
        return tf.nn.embedding_lookup(embeddings, inp)
    
    def mean_nce_loss(self, inp):
        return tf.reduce_mean(tf.nn.nce_loss(weights=self.w,
                                                 biases=self.b,
                                                 labels=self.y,
                                                 inputs=inp,
                                                 num_sampled=self.n_samples,
                                                 num_classes=self.vocab_size), name='loss')
    
    def normalize_embeddings(self):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        normalized_embeddings = self.embeddings / norm
        self.normalized_embeddings = normalized_embeddings
    
    def get_similarity(self, t):
        pred_embeddings = self.embed(self.normalized_embeddings, t)
        sim = tf.matmul(pred_embeddings, self.normalized_embeddings, transpose_b=True)
        return sim
    

    def init_model(self):
        # Input data.
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size], name='x')
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='y')
        self.val_x = tf.constant(self.val_data, name='val_x')
        # embedding var

        with tf.device('/cpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dims], -1.0, 1.0))
            # nce loss weights and biases
            self.w = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_dims],
                                                    stddev=1.0 / math.sqrt(self.embedding_dims)))
            self.b = tf.Variable(tf.zeros([self.vocab_size]))

        embedded = self.embed(self.embeddings, self.x)
        self.loss = self.mean_nce_loss(embedded)
        self.optimize = self.optimizer(self.lr).minimize(self.loss, name='optimize')

        self.normalize_embeddings()
        self.similarity = self.get_similarity(self.val_x)
        self.init_op = tf.global_variables_initializer
        self.saver = tf.train.Saver
