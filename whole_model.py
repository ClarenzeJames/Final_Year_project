import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB0, EfficientNetB3, ResNet50, VGG16
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.metrics import AUC, Recall, SpecificityAtSensitivity
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
import os
import pandas as pd
import re

from evaluation import evaluate_multiclass, plot_history
import patch_model
PatchModel = patch_model.Patch_model()

class Whole_model():
    def __init__(self):
        self.patch_model_save_file = "./best_hybrid_patch_model.weights.h5"
        self.whole_model_save_file = "./best_hybrid_whole_model.weights.h5"
        
        self.batch_size = 4
        self.epochs = 50
        self.patch_size = (224,224)
        self.whole_size = (384, 384)
        self.metrics = [
            'accuracy',
            AUC(name='auc'),
        ]

    def build_whole_model(self):
        input = layers.Input(shape=(*self.whole_size, 3))

        backbone = EfficientNetB3(
            include_top=False,
            weights=None,
            input_shape=(*self.whole_size,3),
            input_tensor=input,
            pooling=None
        )
        
        x = backbone.output
        x = GlobalAveragePooling2D(name='gap')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation='sigmoid', name='exam_out')(x)

        model = Model(inputs=input, outputs=out, name='whole_model')        

        return model
    
    def compile_model(self):
        train_gen, val_gen, test_gen = self.get_generators();

        whole_model = self.build_whole_model()

        patch_model = PatchModel.build_patch_model()
        patch_model.load_weights(self.patch_model_save_file)

        self.trf_backbone_from_patch(patch_model, whole_model)

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=4,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                self.whole_model_save_file,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]

        counts = np.bincount(train_gen.classes)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(counts)),
            y=train_gen.classes
        )

        class_weights = dict(enumerate(class_weights))

        # Stage 1: last layer only
        print("----- Stage 1: Training last layer only -----")
        for layer in whole_model.layers:
            layer.trainable = False
        whole_model.get_layer('exam_out').trainable = True

        whole_model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        history1 = whole_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # Stage 2: Unfreezing top~40 layers
        for layer in whole_model.layers[-40:]:
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        for layer in whole_model.layers[:-40]:
            layer.trainable = False

        whole_model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        history2 = whole_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=200,
            class_weight=class_weights,
            callbacks=callbacks
        )

        #Stage 3: fine-tune all layers
        # for layer in two_view.layers:
        #     layer.trainable = True
        #     if isinstance(layer, BatchNormalization):
        #         layer.trainable = False
        
        # for layer in two_view.layers:
        #     layer.trainable = True
        #     if isinstance(layer, BatchNormalization):
        #         layer.trainable = False

        # two_view.compile(
        #     optimizer=Adam(learning_rate=5e-6),
        #     loss=BinaryCrossentropy(label_smoothing=0.0),
        #     metrics=['accuracy', AUC(name='auc')]
        # )

        # history3 = two_view.fit(
        #     train_gen,
        #     validation_data=val_gen,
        #     epochs=200,
        #     class_weight=class_weights,
        #     callbacks=callbacks
        # )

        plot_history(history1, type="whole", name="whole image stage 1")
        plot_history(history2, type="whole", name="whole image stage 2")



    def compute_class_weights(self,train_df):
        y = train_df[train_df['split'] == 'train']['label'].values.astype(int)
        if len(y) == 0:
            return None
        uniq, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        weights = {int(c): float(total/(len(uniq)*cnt)) for c, cnt in zip(uniq,counts)}
        print(weights)
        return weights
    
    def evaluate_on_test(self):
        _, _, test_gen = self.get_generators()

        model = self.build_whole_model()
        model.load_weights(self.whole_model_save_file)

        model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        results = model.evaluate(test_gen, verbose=1)
        print("Test results:")
        for metric, value in zip(model.metrics_names, results):
            print(f"{metric}: {value:.4f}")
    




    # ----------------------------------------------------
    # ---------------- Transfer functions ----------------
    # ----------------------------------------------------

    def _find_backbone(self,model, name='efficientnetb3'):
        try:
            return model.get_layer(name)
        except Exception:
            pass
        
        for l in model.layers:
            if hasattr(l,"layers") and name in l.name.lower():
                return l
        return None
    
    def trf_backbone_from_patch(self, patch_model, whole_model):
        pmap = {l.name: l for l in patch_model.layers if l.weights}
        copied = 0
        for wl in whole_model.layers:
            if wl.weights and wl.name in pmap and pmap[wl.name].get_weights():
                try:
                    wl.set_weights(pmap[wl.name].get_weights())
                    copied += 1
                except Exception:
                    pass

        print(f"[transfer:fallback] copied {copied} layers by global names")

    def get_generators(self):
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
            f"../../archive/full mammogram images/train",
            target_size = self.patch_size,
            batch_size = self.batch_size,
            class_mode = 'binary',
            shuffle = True
        )

        val_gen = val_datagen.flow_from_directory(
            f"../../archive/full mammogram images/train",
            target_size = self.patch_size,
            batch_size = self.batch_size,
            class_mode = 'binary',
            shuffle = False
        )

        test_gen = val_datagen.flow_from_directory(
            f"../../archive/full mammogram images/test",
            target_size = (384,384),
            batch_size = self.batch_size,
            class_mode = 'binary',
            shuffle = False
        )

        return train_gen, val_gen, test_gen
    
    # ----------------------------------------------------
    # ---------------- K-fold Cross-valid ----------------
    # ----------------------------------------------------
    
    def scan_split(self, root, split_name):
        split_dir = os.path.join(root, split_name)
        rows = []
        for cls in ['benign', 'malignant']:
            cdir = os.path.join(split_dir, cls)
            print(cdir)
            if not os.path.isdir(cdir): continue
            for fn in os.listdir(cdir + "/"):
                print(fn)
                if fn.lower().endswith((".jpg")):
                    path = os.path.join(cdir,fn)
                    m = re.search(r'P_(\d+)', fn)
                    pid = m.group(1) if m else os.path.splitext(fn)[0]
                    rows.append({'filepath': path, 'label_name': cls, 'patient_id': pid})

        df = pd.DataFrame(rows)
        classes = sorted(df['label_name'].unique().tolist())
        name2idx = {c: i for i,c in enumerate(classes)}
        df['label'] = df['label_name'].map(name2idx).astype(int)

        return df


    def flow_df(self,df_split, datagen, shuffle, y_col='label_name'):
        classes = ['benign', 'malignant']

        return datagen.flow_from_dataframe(
            dataframe=df_split,
            x_col='filepath',
            y_col = y_col,
            classes = classes,
            target_size = self.whole_size,
            batch_size = self.batch_size,
            shuffle = shuffle,
            class_mode = 'binary'
        )

    def compile_model_kfold(self):
        root = "../../archive/full mammogram images/"
        root_train_df = self.scan_split(root, "train")
        root_val_df = self.scan_split(root, "validation")
        df = pd.concat([root_train_df, root_val_df], ignore_index=True)

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

        X = df['filepath'].values
        y = df['label'].values
        g = df['patient_id'].values

        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups=g)):
            print(f"===== Fold {fold+1} =====")
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            # class weights per fold
            cls = np.array([0,1])
            cw_arr = compute_class_weight('balanced',
                                          classes=cls,
                                          y=train_df['label'].values)
            
            class_weights = {int(c):float(w) for c,w in zip(cls, cw_arr)}

            train_gen = self.flow_df(train_df, train_datagen, shuffle=True)
            val_gen = self.flow_df(val_df, val_datagen, shuffle=False)

            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_auc',
                    factor=0.5,
                    patience=4,
                    mode='max',
                    verbose=1
                ),
                ModelCheckpoint(
                    self.whole_model_save_file,
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            ]

            two_view = self.build_two_view_model()
            two_view.load_weights(self.whole_model_save_file)
            for layer in two_view.layers:
                layer.trainable = False
            two_view.get_layer('exam_out').trainable = True
            two_view.compile(
                optimizer=Adam(learning_rate=1e-2),
                loss=BinaryCrossentropy(label_smoothing=0.0),
                metrics=['accuracy', AUC(name='auc')]
            )
            history1 = two_view.fit(
                train_gen,
                validation_data=val_gen,
                epochs=10,
                class_weight=class_weights,
                callbacks=callbacks
            )

            # Stage 2: Unfreezing top~40 layers
            for layer in two_view.layers[-40:]:
                layer.trainable = True
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
            for layer in two_view.layers[:-40]:
                layer.trainable = False

            two_view.compile(
                optimizer=Adam(learning_rate=1e-2),
                loss=BinaryCrossentropy(label_smoothing=0.0),
                metrics=['accuracy', AUC(name='auc')]
            )
            if (fold+1) < 2:
                history2 = two_view.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=20,
                    class_weight=class_weights,
                    callbacks=callbacks
                )
            else:
                history2 = two_view.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=100,
                    class_weight=class_weights,
                    callbacks=callbacks
                )

            scores = two_view.evaluate(val_gen, verbose=0)
            print(dict(zip(two_view.metrics_names, [float(s) for s in scores])))
            fold_results.append(dict(zip(two_view.metrics_names, [float(s) for s in scores])))

            if (fold+1) == 2:

                plot_history(history1, 'whole', f'stage 1-2 - k-fold{fold+1}')
                plot_history(history2, 'whole', f'stage 2-2 - k-fold{fold+1}')

        print(fold_results)


    # -----------------------------------------------------
    # ---------------- Test set evaluation ----------------
    # -----------------------------------------------------

    def test_set_eval(self):
        _,_, test_gen = self.get_generators();

        model = self.build_whole_model()
        model.load_weights(self.whole_model_save_file)
        model.compile(
            optimizer=Adam(learning_rate=1e-2),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        results = model.evaluate(test_gen, verbose=1)
        print(dict(zip(model.metrics_names, [float(x) for x in results])))

    



    
