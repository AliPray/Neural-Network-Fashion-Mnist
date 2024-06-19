## Machine Learning Models Trained on Fashion-MNIST Dataset

### Single-Layer Model

#### Overview
The single-layer model was designed as a simple neural network with one hidden layer. It was trained on the Fashion-MNIST dataset, which consists of grayscale images of clothing items. The model's architecture included:
- **Input Layer**: Flattened input representing 28x28 pixel images.
- **Hidden Layer**: Single dense layer with customizable parameters.
- **Output Layer**: Softmax activation function for multi-class classification.

#### Training and Evaluation
- **Epochs**: The model was trained over multiple epochs to optimize performance. Experimentation involved varying the number of epochs to find the optimal training duration.
- **Learning Rate**: Different learning rates were tested to observe their impact on convergence speed and model accuracy.
- **Batch Sizes**: Training batches of various sizes were used to balance computational efficiency and model stability.
- **Weights Initialization**: Initialization techniques such as random or predefined weights were explored to assess their influence on training convergence.
- **Activation Function**: Utilized softmax activation in the output layer for probability distribution across classes.

### Multilayer Model

#### Overview
The multilayer model expanded on the single-layer approach by incorporating multiple hidden layers. This architecture aimed to capture more complex relationships within the Fashion-MNIST dataset.
- **Input Layer**: Same as the single-layer model.
- **Hidden Layers**: Multiple dense layers stacked sequentially, each potentially employing different activation functions (e.g., ReLU).
- **Output Layer**: Softmax activation to classify the input image into one of the ten fashion categories.

#### Training and Evaluation
- **Epochs**: Similar to the single-layer model, training epochs were adjusted to optimize accuracy without overfitting.
- **Learning Rate**: Continued experimentation with learning rates, especially crucial in deeper networks to control gradient descent.
- **Batch Sizes**: Batch size variations were tested to balance memory usage and training speed.
- **Number of Layers**: Explored different depths of the neural network to assess trade-offs between model complexity and performance.
- **Activation Functions**: ReLU activation in hidden layers was tested to introduce non-linearity and improve model expressiveness.
