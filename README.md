# Federated Learning for Chest X-Ray Classification

This project demonstrates the application of **Federated Learning** using a ResNet-34 model to classify chest X-ray images into multiple disease categories. The implementation uses **PyTorch** and **Torchvision**, and is designed to run in a Kaggle notebook environment.

---

## Project Overview

### 🎯 Objective
The goal of this project is to apply federated learning to train a neural network on chest X-ray images **without centralizing data**. This ensures patient privacy while enabling collaborative model training across multiple clients.

### 🗂️ Dataset
The dataset used is the **NIH Chest X-ray Dataset**, which contains over 100,000 chest X-ray images labeled with 14 thoracic diseases. Images and metadata are loaded from the Kaggle input directory, and labels are processed for multi-label classification.

---

## 🔧 How It Works

1. **Load Data**
   - List files in the input directory and locate relevant image folders and metadata.

2. **Preprocess Dataset**
   - Load and explore the CSV file containing image-label metadata.
   - Split data and apply image transformations (resizing, normalization).

3. **Custom Dataset and DataLoader**
   - Define a `Dataset` class for loading images and labels.
   - Use `DataLoader` to enable efficient batching and shuffling.

4. **Model Architecture**
   - Load a pre-trained ResNet-34 model.
   - Replace the final classification layer with a custom multi-label output layer.

5. **Federated Client and Server Setup**
   - Create a `Client` class to handle local model updates.
   - Track only certain layers (e.g., `layer4` and `fc`) for training.
   - Use weighted averaging to update the global model (FedAvg).

6. **Train Federated Model**
   - Split the dataset among clients.
   - Simulate federated rounds where each client trains locally and shares updates.

7. **Monitor Performance**
   - Record training and validation loss at each round.
   - Visualize learning curves.

   ![Round 8–10 Training](https://github.com/Arjun-08/Federated-learning-over-IOMT/assets/88790572/d4c49d27-e52c-446d-957d-203314e31a61)

---

## 📈 Results

The model was trained using federated learning across multiple simulated clients. Performance was monitored using training and validation loss:

![Training History](https://github.com/Arjun-08/Federated-learning-over-IOMT/assets/88790572/df0870e5-430c-4083-90fb-80befeabcd5f)

---

## ✅ Conclusion

This project showcases a practical implementation of federated learning for **medical image classification**, preserving data privacy while maintaining performance. The approach can be adapted to other domains requiring decentralized model training.

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out:  
📧 [nvarjunmani07@gmail.com](mailto:nvarjunmani07@gmail.com)
