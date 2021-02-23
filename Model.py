def create_model():

    VGG16 = keras.applications.VGG16(include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=(8, 8, 512)))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    for layer in VGG16.layers:
        model.add(layer)
        if "conv" in layer.name:
            model.add(BatchNormalization())
    for layer in top_model.layers:
        model.add(layer)
        if "dense" in layer.name:
            model.add(BatchNormalization())

    model.summary()
    return model
