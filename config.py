class config:
    learning_rate = 0.01
    batch_size = 512
    train_idx = 315218
    emb_sz = 100
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    sparse = True
