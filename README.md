Image Captioning using Graph Neural Networks (GCN and GAT)
1. Dataset Source
This project uses the Flickr8k dataset for training and evaluating the image captioning models. The dataset contains 8,000 images, each annotated with 5 human-generated captions.
    • Dataset URL: https://www.kaggle.com/datasets/adityajn105/flickr8k
Steps to Access the Dataset in Google Colab
    1 Visit the Kaggle website and go to your account settings.
    2 In the API section, click on "Create New API Token".
    3 A file named kaggle.json will be downloaded to your system.
    4 In the first cell of your Colab notebook:
        ◦ Upload the kaggle.json file.
        ◦ Set up the Kaggle CLI by moving the token to the proper directory.
        ◦ Install the required packages and download the dataset using Kaggle CLI.
        ◦ Unzip the downloaded dataset into a working directory.

2. Software and Hardware Requirements
Software Requirements
Ensure the following packages are installed before running the project:
    • Python 3.8 or higher
    • PyTorch 2.0 or higher (with CUDA support if GPU is used)
    • Torchvision
    • NLTK
    • scikit-learn
    • Matplotlib
    • NumPy
    • tqdm
Hardware Requirements
    • Recommended Cloud GPU: Google Colab Pro with Tesla T4 or better
    • Recommended Local GPU: NVIDIA GPU with a minimum of 8GB VRAM
    • Minimum Storage: 5GB for storing dataset and model checkpoints

3. Instructions to Execute the Source Code
This project is implemented in a single Python file.
Step 1: Dataset Setup
Download and extract the Flickr8k dataset using Kaggle API as described above. Ensure all files are in the correct directory structure.
Step 2: Feature Extraction
The pre-trained convolutional neural network (ResNet-50) to extract features from each image. These features will be used as node inputs for the graph-based models.
Step 3: Vocabulary Construction
Tokenize the captions and build a vocabulary. Words that occur below a certain frequency threshold are excluded to maintain a compact and meaningful vocabulary.
Step 4: Data Preparation
Constructing the training dataset using the extracted image features and tokenized captions. By applying batching, padding, and masking where required for model input preparation.
Step 5: Model Architecture
Two models are implemented for comparison:
    • A GCN-based model that uses graph convolutions to learn spatial and semantic relationships between image regions.
    • A GAT-based model that applies graph attention mechanisms for weighted feature propagation based on importance.
Step 6: Training the Model
Train both models using cross-entropy loss. Appropriate optimizers and learning rate schedules are used. Evaluation is performed after each epoch to monitor performance.
Step 7: Evaluation
Evaluate the trained models on a separate test set using:
    • BLEU Score for n-gram precision-based caption matching.
    • ROUGE Score for recall-based assessment of generated captions against references.
These metrics provide quantitative insights into the quality and relevance of generated image captions.
