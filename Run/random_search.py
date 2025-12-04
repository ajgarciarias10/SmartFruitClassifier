import random
import numpy as np
from transfer_model import TransferLearningModel
# =======
# Search Space
# =======
def sample_hyparams():
    return {
        "dense_units":random.choice([256, 512, 1024]),
        "dropout_rate":random.choice([0.3, 0.4 ,0.5, 0.6, 0.7]),
        "learning_rate": 10 ** np.random.uniform(-5,-3),
        "batch_size": random.choice([16, 32, 64])
    }
    
# =======
# Random Search
# =======   
def random_search( train_dir,
    val_dir,
    img_size,
    num_classes,
    n_trials,
    epochs,
    augmentation):

    BEST_MODEL_PATH = "best_random_search_model.keras"
    best_val_acc = 0.0
    best_hyparams = None

    for trial in range(n_trials):
        print(f"\n========== Trial {trial+1}/{n_trials} ==========")
        
        hp= sample_hyparams()
        print(f"Hyperparameters: {hp}")
        
        
        model_obj = TransferLearningModel(img_size=img_size, num_classes=num_classes)
        model = model_obj.build_model(
                learning_rate=hp["learning_rate"],
                dense_units=hp["dense_units"],
                dropout_rate=hp["dropout_rate"],
                freeze_base=True
        )
        train_gen, val_gen = model_obj.create_data_generators(
                train_dir=train_dir,
                val_dir=val_dir,
                batch_size=hp["batch_size"],
                augmentation=augmentation
        )
        
        history = model_obj.train(train_gen, val_gen, epochs=epochs)
        
        val_acc = max(history.history['val_accuracy']) 
        print(f"Best accuracy val: {val_acc:.4f}")  
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_hyparams = hp
            
            model_obj.model.save(BEST_MODEL_PATH)
            print(f"New best model saved with val accuracy: {best_val_acc:.4f}")
            
    print("\n========== Random Search Complete ==========")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Hyperparameters: {best_hyparams}")
    return best_hyparams
