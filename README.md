# Real-Time Image Processing

## Description
A real-time image processing web application utilizing parallel processing techniques. This project demonstrates the use of multithreading and multiprocessing for efficient image filtering operations such as grayscale, blur, edge detection, sharpening, and sepia. Developed using Python, Flask, OpenCV, and Flask-SocketIO.

## Features
- Upload images and apply various filters in real-time.
- Filters available: grayscale, blur, edge detection, sharpening, and sepia.
- Utilizes multithreading and multiprocessing for efficient image processing.
- Real-time updates using WebSocket (Flask-SocketIO).

## files structure
1)app.py
2)templates (folder)--> (index.html),(result.html)                                                                                                                                                                                          
                

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/real-time-image-processing.git
    cd real-time-image-processing
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the application:
    ```sh
    python app.py
    ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the application.

## Usage
1. Select an image file to upload.
2. Choose a filter to apply from the available options.
3. View the processed image in real-time.

## Contributing
Contributions are welcome! Please create a pull request or open an issue for any improvements or suggestions.

## License
This project is licensed under the MIT License.
