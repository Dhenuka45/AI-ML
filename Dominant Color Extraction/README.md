#Dominant Color Extraction using K-Means Clustering

This project uses K-Means Clustering to extract the most dominant colors from an image. It’s a simple yet powerful example of how unsupervised machine learning can be applied to image processing and computer vision tasks.

# Features

Upload any image and get the top N dominant colors
Visualizes the color palette using matplotlib
Can be used for color theme generation, design inspiration.

# Technologies Used

Python 
OpenCV
scikit-learn
NumPy
Matplotlib

# How It Works

The image is loaded and converted from BGR to RGB.
It is reshaped into a 2D array of pixels.
K-Means clustering is applied to group pixels into k clusters.
The cluster centers (centroids) are interpreted as dominant RGB colors.
The result is visualized as 'K' color bars or swatches.

Input Image:

<img width="920" height="518" alt="image" src="https://github.com/user-attachments/assets/811e421c-a235-455c-86b0-2aa7c05e72fa" />

Output:

<img width="962" height="263" alt="image" src="https://github.com/user-attachments/assets/dd4ba518-bb69-431f-a83d-2ef5852cd470" />


# Contributing

Feel free to fork this repo and open a pull request if you’d like to improve it or add new features.

# License

This project is licensed under the MIT License.

# Author

Dhenuka Dudde (https://github.com/Dhenuka45)
