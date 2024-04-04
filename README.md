I have stained a colon with 3D H&E and imaged with a confocal microscope. I wanted to crop and augment the data for training neural networks later.

I came up with a toy problem to practice my coding skills first. I trained a simple NN to classify horizontal or vertical colon images (since I produced horizontal and vertical colon images during data augmentation). 
The task is pretty simple. After 5 epochs of training, the model performs 100% accuracy.

In this folder, there are two example images. The train.py file can be used for training, saving the model and evaluating the model directly. The train (first epoch, 20 batches only).py file is used for early termination to see the accuracy with minimal training.
