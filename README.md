
# Emotional Dynamics Analysis

## Overview
This repository contains code and datasets aimed at modeling emotional dynamics such as love and friendship over time using theoretical formulas and machine learning techniques. The models are built using Python and leverage various data science and machine learning libraries to predict and understand how emotional metrics evolve.

## Project Structure
- `models/`: Contains Python scripts for different regression models.
- `data/`: Includes datasets used in the models, formatted in Excel sheets.
- `results/`: Stores output plots and model analysis results.
- `README.md`: Provides an overview and usage instructions for the project.

## Formulas Used
The project uses a set of creatively devised formulas to model the complexities of emotions such as love and friendship over time. These are not empirically derived but serve as illustrative mathematical representations of how such emotions might be quantified and analyzed.

### Love Over Time
\[ L(t) = a \cdot e^{r \cdot t} + b \cdot \sin(c \cdot t + d) \]
Where:
- \( L(t) \) is the level of love at time \( t \).
- \( a \), \( r \) adjust the exponential growth rate.
- \( b \), \( c \), \( d \) control the oscillatory behavior, representing emotional fluctuations.

### Friendship Over Time
\[ F(t) = a \cdot \log(b \cdot t + 1) + c \cdot \cos(d \cdot t + e) \]
Where:
- \( F(t) \) represents the level of friendship at time \( t \).
- \( a \), \( b \) dictate the logarithmic increase.
- \( c \), \( d \), \( e \) manage the amplitude and frequency of cyclical changes.

### Depth of Friendship as a Function of Depth of Love
\[ D(f) = a \cdot \sqrt{f} + b \cdot \sin(c \cdot f + d) \]
Where:
- \( D(f) \) indicates the depth of friendship given the depth of love \( f \).
- \( a \) modifies the primary growth rate.
- \( b \), \( c \), \( d \) influence periodic variations.

## Data Description
The datasets are generated to simulate the behavior of these emotional dynamics, with each dataset containing 500 data points spread across three repetitions to mimic experimental variability.

## Usage
To use the models, ensure you have Python installed along with necessary libraries like Pandas, Matplotlib, Scikit-learn, etc. Follow the instructions in the individual model scripts for more detailed steps on execution and analysis.

## Contributing
Contributions to the project are welcome. Please ensure to follow the existing code structure and submit pull requests for any enhancements, bug fixes, or additional models.

## License
This project is available under the MIT License. See the LICENSE file for more details.
