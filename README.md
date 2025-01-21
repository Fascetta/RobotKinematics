# Machine Learning Approaches to Robot Kinematics

## Overview

This project explores machine learning methods to solve forward and inverse kinematics for robotic arms of varying configurations. It evaluates the performance of multiple models—Feedforward Neural Networks (FNN), Support Vector Regression (SVR), and k-Nearest Neighbors (KNN)—in predicting the end-effector position given joint angles. Data was generated using the **MuJoCo** simulation framework for three robot configurations: 
- **2D Two-Joint Arm**
- **2D Three-Joint Arm**
- **3D Five-Joint Arm (MARRtino)**

Key achievements include:
- Learning and validating forward kinematics using machine learning models.
- Computing and comparing Jacobian matrices derived from models with analytical solutions.
- Solving inverse kinematics using the Newton-Raphson method.

---

## Objectives

1. Generate and preprocess datasets for robot configurations.
2. Train models to predict end-effector positions using joint angles.
3. Evaluate model accuracy and computational efficiency.
4. Validate learned Jacobians against analytical ones for 2D configurations.
5. Solve inverse kinematics using the learned Jacobians.

---

## Implemented Solutions

### Data Generation
- **Environment**: MuJoCo simulation.
- **Robots**: Configurations with 2, 3, and 5 degrees of freedom (DOF).
- **Datasets**:
  - Generated with random seeds (1, 10, 100).
  - Steps: 1,000 to 100,000.
- **Features**: Joint angles (`θ1, θ2, ..., θn`).
- **Labels**: End-effector positions (`x, y, z`).

### Models
1. **Feedforward Neural Network (FNN)**
   - Framework: TensorFlow/Keras.
   - Architecture: 3 layers (128, 112, 80 units).
   - Optimizer: Adam with learning rate scheduling.
   - Early stopping to prevent overfitting.

2. **Support Vector Regression (SVR)**
   - Kernel: Radial Basis Function (RBF).
   - MultiOutputRegressor for multidimensional predictions.
   - Key Parameters:
     - C: 50.0
     - Epsilon: 1e-4

3. **k-Nearest Neighbors (KNN)**
   - Distance Metrics: Euclidean.
   - Number of neighbors: 3.

---

## Results

### Performance Metrics
| Model | Mean Squared Error (MSE) | Training Time | Inference Time |
|-------|---------------------------|---------------|----------------|
| FNN   | 8.65e-06                 | 8m34s         | 2.7s           |
| SVR   | 1.69e-07                 | 15s           | 0.1s           |
| KNN   | 6.45e-05                 | 0s            | 0.1s           |

### Jacobian Comparison
- **Learned Jacobians** from FNN models approximated the **analytical Jacobians** accurately for the 2D robot.
- Example:
  - Analytical: `[[0.004, 0.099], [0.04, 0.011]]`
  - Learned: `[[0.0038, 0.098], [0.038, 0.01]]`

### Inverse Kinematics
- **Method**: Newton-Raphson algorithm.
- **Result**: Reliable convergence to target positions.
  - Example:
    - Target Position: `[-0.058, -0.087]`
    - Predicted Joint Angles: `[12.789, -14.101]`
    - Converged in 8 iterations.

---

## Insights

- **SVR** demonstrated the best overall accuracy and robustness.
- **FNN** required precise tuning but showed potential for scalability.
- **KNN** offered simplicity but lacked effectiveness in higher dimensions.

---

## Future Work

1. **PID Controller**: Implement a controller to achieve precise joint configurations.
2. **Deep Reinforcement Learning**: Train policies for target positions using reinforcement learning.
3. **Extended Evaluation**: Explore additional robot configurations and compare alternative machine learning methods.

---

## Dependencies

- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- MuJoCo

---

## Citation

If you reference this work, please use the following format:

```bibtex
@misc{bianchi2024robotkinematics,
  title={Machine Learning Approaches to Robot Kinematics},
  author={Christian Bianchi},
  year={2024}
}
