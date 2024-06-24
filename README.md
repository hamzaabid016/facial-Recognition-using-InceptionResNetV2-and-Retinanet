# Facial Recognition Project

This project is a facial recognition system designed to identify and verify individuals using their facial features. It includes a web application and tools for managing a facial recognition database.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Facial recognition using deep learning models
- Web interface for adding and recognizing faces
- Database integration for storing recognized faces and user data
- Docker support for easy deployment

## Installation

To set up the project locally, follow these steps:

### Prerequisites

- Python 3.7 or higher
- Docker (optional, for containerized deployment)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/hamzaabid016/Facial_recognition-main.git
    cd Facial_recognition-main
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the database:**

    ```bash
    python app.py db init
    python app.py db migrate
    python app.py db upgrade
    ```

5. **Run the application:**

    ```bash
    python app.py
    ```


