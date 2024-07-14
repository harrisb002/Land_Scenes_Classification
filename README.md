# Classification of Land Uses and Covers from Satellite Imagery

## Overview
This project aims to create a neural network that classifies land uses and land covers from satellite imagery using the EuroSat dataset. It involves saving and loading models with TensorFlow callbacks and comparing the performance with a pre-trained neural network classifier.

## Table of Contents
- [Getting Started](#getting-started)
- [Neural Network Creation](#neural-network-creation)
- [Model Saving and Loading](#model-saving-and-loading)
- [Performance Comparison](#performance-comparison)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Credits](#credits)

## Getting Started
### Instructions
This project will guide you through creating a neural network for classifying land uses and land covers from satellite imagery. The EuroSat dataset is used, and the model is saved using TensorFlow callbacks for future use.

### Setup
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook Classification.ipynb
   ```

## Neural Network Creation
Develop a neural network to classify satellite images into different land use and cover categories. The detailed implementation is available in the Jupyter notebook.

## Model Saving and Loading
Utilize TensorFlow callbacks to save the model during training and reload it later for evaluation and further use. This demonstrates the practical steps in model persistence and transfer learning.

## Performance Comparison
Load a pre-trained neural network classifier and compare its performance with the newly created model. This helps in understanding the effectiveness of different models on the same dataset.

## Technologies Used
- **TensorFlow:** For building and training the neural network.
- **EuroSat Dataset:** For satellite imagery.
- **Python:** For scripting and implementation.

## Usage
The project demonstrates practical steps in saving and loading machine learning models, enhancing the understanding of model persistence and transfer learning in the context of satellite image classification.

## Credits
Some of the material in this notebook has been taken from a Coursera Deep Learning course.

---

For more details, please refer to the [Jupyter notebook](Classification.ipynb).