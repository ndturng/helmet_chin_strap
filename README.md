# helmet_chin_strap
Training model to distinguish workers wearing helmets with straps or not. 
## What i do
Using k-Fold Cross-Validation to evaluate and select the optimal model (here we chose RepVGG). Also, I use cross-validation to check and select the optimizer cũng như learning rate scheduler. Here I chose the Adam optimizer and StepLR scheduler. 

Using model **RepVGG** (pretrained ImageNet) as a **Feature Extractor (FE)**, customize the last Fully Connected layer to serve the 3-class classification problem *(no_chin_strap, with_chin_strap and undetermined)*. Use the learning rate scheduler to improve accuracy.
## What i got



Download model 50 epochs pretrained here: https://drive.google.com/file/d/1QD00NIetpCGKnkgEGjnY5wUP5xymwLim/view?usp=sharing
