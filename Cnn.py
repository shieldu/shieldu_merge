from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNN 모델 구성
def train_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # 고양이(0) / 강아지(1)
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 학습 데이터 준비
    train_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory('/Users/kjh/kjh_Project/AI/MP3/cat_dog', target_size=(64, 64), batch_size=32, class_mode='binary')

    # 모델 학습
    model.fit(training_set, steps_per_epoch=8000, epochs=5, validation_steps=2000)

    # 모델 저장
    model.save('/Users/kjh/kjh_Project/AI/MP3/cnn_model.h5')

train_cnn()