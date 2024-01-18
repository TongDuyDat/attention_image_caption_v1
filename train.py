from dataloder import data_train_test_val

train_loader, val_loader, test_loader = data_train_test_val(8)



for data in train_loader:
    images, caption = data
    print(images.shape, caption.shape)