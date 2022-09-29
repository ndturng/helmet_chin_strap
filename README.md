# helmet_chin_strap
Training model to distinguish workers wearing helmets with straps or not. Using k-Fold Cross-Validation to evaluate and select the optimal model (RepVGG). Using model RepVGG (pretrained ImageNet) as a Feature Extractor (FE), customize the last Fully Connected layer to serve the 3-class classification problem (no_chin_strap, with_chin_strap and undetermined). Use the learning rate scheduler to improve accuracy.
Download model pretrained here: https://drive.google.com/file/d/1QD00NIetpCGKnkgEGjnY5wUP5xymwLim/view?usp=sharing
