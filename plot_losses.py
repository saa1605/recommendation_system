import matplotlib.pyplot as plt
plt.subplot(2, 2, 1)

plt.plot(np.arange(len(train_losses)),train_losses)

plt.xlabel("Epochs")
plt.ylabel("RMSE Loss")
plt.title('Matrix Factorization Train')


plt.subplot(2, 2, 2)
plt.plot(np.arange(len(test_losses)),test_losses)

plt.xlabel("Epochs")
plt.ylabel("RMSE Loss")
plt.title('Matrix Factorization Test')

plt.subplot(2, 2, 3)
plt.plot(np.arange(len(train_losses_nn)),train_losses_nn)

plt.xlabel("Epochs")
plt.ylabel("RMSE Loss")
plt.title('Neural Network Train')

plt.subplot(2, 2, 4)
plt.plot(np.arange(len(test_losses_nn)),test_losses_nn)

plt.xlabel("Epochs")
plt.ylabel("RMSE Loss")
plt.title('Neural Network Test')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.6)