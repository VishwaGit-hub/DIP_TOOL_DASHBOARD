How to Run the Project

Follow the steps below to set up the environment and run the application on your system.

1. Download the Project Files
-Download the project folder (ZIP).
-Extract it to any location on your computer.
-Ensure the folder contains:
-The main Python file (e.g., DIP_TOOLS.py)
-Any sample images (optional)
-requirements.txt

2. Create a Virtual Environment
-Windows
python -m venv venv
venv\Scripts\activate

-macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

-Run the following inside the extracted folder:
pip install -r requirements.txt


4. Run the Application
-python image_processing_tool.py
(Replace the filename if your main file uses a different name.)

5. Close & Deactivate (Optional)
-When you're done:
deactivate