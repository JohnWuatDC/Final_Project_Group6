def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(GRU_UNITS, return_sequences=True))(x) 
    x = Bidirectional(CuDNNGRU(GRU_UNITS, return_sequences=True))(x) 
