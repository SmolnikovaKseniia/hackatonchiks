def predict(images):
    pred = model.predict(image, verbose=0)[0]
    pred = np.argmax(pred, axis=-1)
    pred = pred.astype(np.int32)
    return pred