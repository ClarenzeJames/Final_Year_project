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
# from tensorflow.keras.applications.vgg16 import preprocess_input #! This is wrong
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.metrics import AUC, Recall, SpecificityAtSensitivity
from sklearn.utils.class_weight import compute_class_weight

from evaluation import evaluate_multiclass, plot_history

"TypeError: Unable to serialize [2.0896919 2.1128857 2.1081853] to JSON. Unrecognized type <class 'tensorflow.python.framework.ops.EagerTensor'>."

class HyrbidModel():
    def __init__(self):
        # self.patch_model_save_file = "./best_hybrid_patch_model.h5"
        self.patch_model_save_file = "./best_hybrid_patch_model.weights.h5"
        self.whole_model_save_file = "./best_hybrid_whole_model.weights.h5"
        
        self.batch_size = 4
        self.epochs = 50
        self.learning_rate = 1e-4
        self.patch_size = (224,224)
        self.whole_size = (384, 384)
        self.metrics = [
            'accuracy',
            AUC(name='auc'),
        ]

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
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(5, activation='softmax', name='patch_out')(x)

        model = Model(inputs=input, outputs=output)
        return model
    
    def compile_patch_model(self):
        # train_gen, val_gen, test_gen = self.get_generators('cropped images')
        train_gen, val_gen = self.get_patch_dataset()
        model = self.build_patch_model()
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
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

        # Stage 1: only last layer
        for layer in model.layers:
            layer.trainable = False


        model.get_layer('patch_out').trainable = True
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks
        )

        # Stage 2: fine-tune last 20 layers
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-20:]:
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks
        )
        # Stage 3: fine-tune all layers
        for layer in model.layers:
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history3 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=callbacks
        )

        # model.save(self.patch_model_save_file, include_optimizer=False)

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

        plot_history(history1,'patch')
        plot_history(history2,'patch')
        plot_history(history3,'patch')

        return history1, history2, history3

    def make_two_view_df(self,df, split):
        df = df[df['split'] == split].reset_index(drop=True)

        cc = tf.constant(df['cc_image_path'].values)
        mlo = tf.constant(df['mlo_image_path'].values)
        y = tf.constant(df['label'].values.astype(np.float32))

        ds = tf.data.Dataset.from_tensor_slices((cc, mlo, y))

        AUG = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.04),
                layers.RandomZoom(0.15),
                layers.RandomTranslation(0.05, 0.05),
                layers.RandomBrightness(0.2),
                # layers.RandomShear(0.05,0.05)
            ],name="aug_whole")

        def _load_and_preprocess(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.whole_size)
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img)
            return img

        def _load(cc_path, mlo_path, label):
            cc_img = _load_and_preprocess(cc_path)
            mlo_img = _load_and_preprocess(mlo_path)
            return {"cc":cc_img, "mlo":mlo_img}, tf.reshape(label,(1,))
        
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        if split == 'train':
            ds = ds.shuffle(2, reshuffle_each_iteration=True).map(lambda x,y: ({"cc":AUG(x['cc'], training=True), "mlo":AUG(x['mlo'])}, y),
                                                                     num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(2).prefetch(tf.data.AUTOTUNE)
        return ds, len(df)
    
    def build_two_view_model(self):
        backbone = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.whole_size, 3),
            pooling='avg'
        )

        cc_in = layers.Input(shape=(*self.whole_size, 3), name='cc')
        mlo_in = layers.Input(shape=(*self.whole_size, 3), name='mlo')

        cc_feat = backbone(cc_in)
        mlo_feat = backbone(mlo_in)

        fused = layers.Concatenate()([cc_feat, mlo_feat])
        fused = layers.Dropout(0.5)(fused)
        fused = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="fc_fused")(fused)
        fused = layers.Dropout(0.5)(fused)
        out = layers.Dense(1, activation='sigmoid', dtype="float32", name='exam_out')(fused)

        model = Model(inputs=[cc_in, mlo_in], outputs=out, name='two_view_model')

        return model
    
    def transfer_backbone_from_patch(self, patch_model, two_view_model):
        # patch_backbone = None
        # for l in patch_model.layers:
        #     # print(l.name)
        #     if isinstance(l, tf.keras.Model) or isinstance(l, tf.keras.layers.Layer):
        #         if l.name.startswith('efficientnetb3'):
        #             patch_backbone = l; break

        # two_view_backbone = None
        # for l in two_view_model.layers:
        #     if isinstance(l, tf.keras.Model) or isinstance(l, tf.keras.layers.Layer):
        #         if l.name.startswith('efficientnetb3'):
        #             two_view_backbone = l; break

        # assert patch_backbone is not None, "patch backbone not found"
        # assert two_view_backbone is not None, "two view backbone not found"
        # pmap = {l.name: l for l in patch_model.layers}
        # for wl in two_view_model.layers:
        #     if wl.name in pmap and pmap[wl.name].get_weights():
        #         try:
        #             wl.set_weights(pmap[wl.name].get_weights())
        #         except:
        #             pass

        try:
            patch_backbone = patch_model.get_layer('efficientnetb3')
            two_view_backbone = two_view_model.get_layer('efficientnetb3')
        except ValueError:
            print("------- Failed initial transfer, going to global -------")
            return self.transfer_by_global_names(patch_model, two_view_model)
        
        print("------- Starting transfer -------")
        pmap = {l.name: l for l in patch_backbone.layers if l.weights}
        copied = 0
        for wl in two_view_backbone.layers:
            if wl.weights and wl.name in pmap and pmap[wl.name].get_weights():
                try:
                    wl.set_weights(pmap[wl.name].get_weights())
                    copied += 1
                except Exception:
                    pass
        
        print("------- Finished transfer -------")
        print(f"[transfer] copied {copied} layers from efficientnetb3")
        
    def transfer_by_global_names(self, patch_model, two_view_model):
        print("------- Starting global transfer -------")
        pmap = {l.name: l for l in patch_model.layers if l.weights}
        copied = 0
        for wl in two_view_model.layers:
            if wl.weights and wl.name in pmap:
                pw = pmap[wl.name].get_weights()
                if pw:
                    try:
                        wl.set_weights(pw)
                        copied += 1
                    except Exception:
                        pass

        print("------- Finished global transfer -------")
        print(f"[transfer:fallback] copied {copied} layers in global names")
    
    def compile_two_view_model(self, train_df, val_df):
        train_ds, n_train = self.make_two_view_df(train_df, split='train')
        val_ds, n_val = self.make_two_view_df(val_df, split='val')

        two_view = self.build_two_view_model()

        patch_model = self.build_patch_model()
        patch_model.load_weights(self.patch_model_save_file)
        two_view.summary()
        patch_model.summary()
        self.trf_backbone_from_patch(patch_model, two_view)

        # Computing class_weights
        labels = train_df['label'].values
        class_weights_arr = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        class_weights = dict(zip(np.unique(labels), class_weights_arr))


        callbacks = [
            # EarlyStopping(
            #     monitor='val_auc',
            #     patience=8,
            #     restore_best_weights=True,
            #     mode='max',
            #     verbose=1
            # ),
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

        # steps_per_epoch = max(1, n_train // self.batch_size)
        # validation_steps = max(1, n_val // self.batch_size)

        # Stage 1: last layer only
        print("----- Stage 1: Training last layer only -----")
        for layer in two_view.layers: 
            layer.trainable = False
        two_view.get_layer('fc_fused').trainable = True
        two_view.get_layer('exam_out').trainable = True

        two_view.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        history1 = two_view.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            # steps_per_epoch=steps_per_epoch,
            # validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Stage 2: Unfreeze top ~20 layers
        print("----- Stage 2: Fine-tuning top layers -----")
        for layer in two_view.layers[-20:]: 
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        for layer in two_view.layers[:-20]: 
            layer.trainable = False
        
        two_view.compile(
            optimizer=Adam(learning_rate=1e-6),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        history2 = two_view.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            # steps_per_epoch=steps_per_epoch,
            # validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Stage 3: fine-tune all layers
        for layer in two_view.layers: 
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        
        two_view.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=BinaryCrossentropy(label_smoothing=0.0),
            metrics=['accuracy', AUC(name='auc')]
        )

        history3 = two_view.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            # steps_per_epoch=steps_per_epoch,
            # validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights
        )

        plot_history(history1, "whole",'whole_stage_1')
        plot_history(history2, "whole",'whole_stage_2')
        plot_history(history3, "whole",'whole_stage_3')


        

        return two_view, history1, val_ds
    
    # def build_hierarchical_model(self):
    #     input = layers.Input(shape=(*self.patch_size, 3))

    #     base = EfficientNetB3(include_top=False, 
    #                           weights='imagenet', 
    #                           input_tensor=input, 
    #                           pooling='avg')
    #     x = base.output
    #     x = layers.Dropout(0.4)(x)
    #     x = layers.Dense(256, activation='relu', kernel_regularizers=regularizers.l2(1e-4))(x)
    #     x = layers.Dropout(0.4)(x)
    #     # x = GlobalAveragePooling2D(name='gap')(x)
    #     x = layers.Dropout(0.2)(x)
    #     # Head 1: type
    #     type_logits = layers.Dense(3, activation='softmax', name='type_out')(x)
    #     # Head 2
    #     mal_logits = layers.Dense(2, activation='softmax', name='mal_out')(x)
    #     # output = layers.Dense(5, activation='softmax', name='patch_out')(x)

    #     model = Model(inputs=input, outputs=[type_logits,mal_logits],name='hier_patch')
    #     return model




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
            '../../archive/5_classes/train/',
            target_size = self.patch_size,
            batch_size = self.batch_size,
            class_mode = 'sparse',
            # classes = ['neg', 'pos'],
            shuffle = True
        )

        val_gen = val_datagen.flow_from_directory(
            '../../archive/5_classes/validation/',
            target_size = self.patch_size,
            batch_size = self.batch_size,
            class_mode = 'sparse',
            # classes = ['neg', 'pos'],
            shuffle = False
        )

        return train_gen, val_gen

    def get_generators(self,image_type):
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

        if image_type == 'cropped images':
            train_gen = train_datagen.flow_from_directory(
                f"../../archive/patch/{image_type}/train",
                target_size = self.patch_size,
                batch_size = self.batch_size,
                class_mode = 'sparse',
                # classes = ['neg', 'pos'],
                shuffle = True
            )

            val_gen = val_datagen.flow_from_directory(
                f"../../archive/patch/{image_type}/validation",
                target_size = self.patch_size,
                batch_size = self.batch_size,
                class_mode = 'sparse',
                # classes = ['neg', 'pos'],
                shuffle = False
            )

            test_gen = val_datagen.flow_from_directory(
                f"../../archive/patch/{image_type}/test",
                target_size = self.patch_size,
                batch_size = self.batch_size,
                class_mode = 'sparse',
                # classes = ['neg', 'pos'],
                shuffle = False
            )

        return train_gen, val_gen, test_gen
    
    # ------------------------------------------------------------------
    # ----------------------- Transfer functions -----------------------
    # ------------------------------------------------------------------

    def _find_backbone(self,model, name='efficientnetb3'):
        try:
            return model.get_layer(name)
        except Exception:
            pass
        
        for l in model.layers:
            if hasattr(l,"layers") and name in l.name.lower():
                return l
        return None
    
    def trf_backbone_from_patch(self, patch_model, two_view_model):
        # patch_bb = self._find_backbone(patch_model, 'efficientnetb3')
        two_bb = self._find_backbone(two_view_model, 'efficientnetb3')
        # print(patch_bb)
        # print(two_bb)

        # if patch_model is not None and two_bb is not None:
        #     two_bb.set_weights(patch_model.get_weights())
        #     print(f"[transfer] copied {len(patch_model.get_weights())} arrays from {patch_model.name} -> {two_bb.name}")
        #     return
        
        # print("[transfer submodel not found; falling back to name-matching]")
        pmap = {l.name: l for l in patch_model.layers if l.weights}
        copied = 0
        for wl in two_bb.layers:
            if wl.weights and wl.name in pmap and pmap[wl.name].get_weights():
                try:
                    wl.set_weights(pmap[wl.name].get_weights())
                    copied += 1
                except Exception:
                    pass

        print(f"[transfer:fallback] copied {copied} layers by global names")
