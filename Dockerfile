# 1. Base Python Image
 
FROM python:3.12-slim
 
# 2. Set Working Directory
WORKDIR /app

# 3. Copy Dependencies File
COPY requirements.txt .

# 4. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Files
COPY src/ ./src/
COPY models/ ./models/

# 6. Run Prediction Application
CMD ["python", "src/predict.py"]