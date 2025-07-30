import os
import numpy as np
import librosa
from keras import layers, models, regularizers
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

def audio_to_spectrogram(file_path, n_mels=128, max_len=128):
    y, sr = librosa.load(file_path, duration=3.0) 
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  
    if spectrogram.shape[1] < max_len:
        pad_width = max_len - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :max_len]
    spectrogram = np.expand_dims(spectrogram, axis=-1)  
    return spectrogram

def prepare_data(data_dir, n_mels=128, max_len=128):
    data = []
    labels = []
    class_names = os.listdir(data_dir)  
    class_map = {name: i for i, name in enumerate(class_names)}  

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            spectrogram = audio_to_spectrogram(file_path, n_mels=n_mels, max_len=max_len)
            data.append(spectrogram)
            labels.append(class_map[class_name])

    return np.array(data), np.array(labels), class_names


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4), 

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),  
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)  
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

data_dir = "voice_data"  
X, y, class_names = prepare_data(data_dir)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


input_shape = X_train.shape[1:]  
num_classes = len(class_names)


model = build_model(input_shape, num_classes)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=float(1e-6))

model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_test, y_test), callbacks=[reduce_lr])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss} / Test accuracy: {test_accuracy}")

def preprocess_wav(file_path, n_mels=128, max_len=128):
    y, sr = librosa.load(file_path, duration=3.0)  
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)  
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  
    if spectrogram.shape[1] < max_len:
        pad_width = max_len - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :max_len]
    spectrogram = np.expand_dims(spectrogram, axis=-1)  
    return np.array([spectrogram])  



unseen_wavs = [
    'voice_data/Mr.Narayan Murthy/clip_24.wav',
    'voice_data/Mr.Narendra Modi/clip_22.wav',
    'voice_data/Mr.Ratan Tata/clip_24.wav',
    'voice_data/Mr.S Jaishankar/clip_51.wav',
    'voice_data/Mr.Sam Altman/clip_220.wav',
    'voice_data/Mr.Shah Rukh Khan/clip_25.wav',
    'voice_data/Mr.Shashi Tharoor/clip_76.wav',
    'voice_data/Mr.Steve Jobs/clip_126.wav',
    'voice_data/Mr.Vir Das/clip_14.wav',
    'voice_data/Mr.Warren Buffet/clip_29.wav'
]

for wav_file in unseen_wavs:
    X_new = preprocess_wav(wav_file)
    
    prediction = model.predict(X_new)
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class]

    print(f"The predicted class for {wav_file} is: {predicted_class_name}")
