import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB0, EfficientNetB3, ResNet50, VGG16
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.metrics import AUC, Recall, SpecificityAtSensitivity
from sklearn.utils.class_weight import compute_class_weight

from evaluation import evaluate_multiclass, plot_history

class Patch_model():
    def __init__(self):
        self.patch_model_save_file = "./best_hybrid_patch_model.weights.h5"
        self.whole_model_save_file = "./best_hybrid_whole_model.weights.h5"
        
        self.batch_size = 8
        self.epochs = 50
        self.patch_size = (224, 224)

        self.patch_metrics = [
            'accuracy',
        ]

    def build_patch_model(self):
        input = layers.Input(shape=(*self.patch_size, 3))

        base = EfficientNetB3(include_top=False, 
                              weights='imagenet', 
                              input_tensor=input, 
                              pooling=None)
        x = base.output
        x = GlobalAveragePooling2D(name='gap')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(5, activation='softmax', name='patch_out')(x)

        model = Model(inputs=input, outputs=output)
        return model
    
    def compile_patch_model(self):
        train_gen, val_gen = self.get_patch_dataset()
        model = self.build_patch_model()

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=4,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                self.patch_model_save_file,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]

        # Computing class weights
        counts = np.bincount(train_gen.classes)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(counts)),
            y=train_gen.classes
        )

        class_weights = dict(enumerate(class_weights))
        print(class_weights)


        # Stage 1: only last layer
        for layer in model.layers:
            layer.trainable = False
        model.get_layer('patch_out').trainable = True

        model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Stage 2: fine-tune last 20 layers
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-40:]:
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Stage 3: fine-tune all layers
        for layer in model.layers:
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history3 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=25,
            callbacks=callbacks,
            class_weight=class_weights
        )

        class_names = [
            "background",
            "benign_calcification",
            "benign_mass",
            "malignant_calcification",
            "malignant_mass",
        ]

        results = evaluate_multiclass(
            model=model,
            val_ds=val_gen,
            num_classes=5,
            class_names=class_names
        )

        print(results)

        plot_history(history1,'patch','patch_stage_1')
        plot_history(history2,'patch','patch_stage_2')
        plot_history(history3,'patch','patch_stage_3')

        return history1, history3
    
    # ----------------------------------------------------------------
    # ----------------------- Helper functions -----------------------
    # ----------------------------------------------------------------

    def get_patch_dataset(self):
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_gen = train_datagen.flow_from_directory(
            '../../archive/5_classes/cropped images/train/',
            target_size = self.patch_size,
            batch_size = self.batch_size,
            class_mode = 'sparse',
            shuffle = True
        )

        val_gen = val_datagen.flow_from_directory(
            '../../archive/5_classes/cropped images/validation/',
            target_size = self.patch_size,
            batch_size = self.batch_size,
            class_mode = 'sparse',
            shuffle = False
        )

        return train_gen, val_gen