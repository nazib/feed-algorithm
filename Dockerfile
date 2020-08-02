# Python image to use.
FROM python:3.8
FROM tensorflow/tensorflow:latest-py3
# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
#RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install -r requirements.txt 
# Copy the rest of the working directory contents into the container at /app
COPY . .

EXPOSE 5000
# Start the server when the container launches
#CMD ["python", "-m", "ptvsd", "--port", "3000", "--host", "0.0.0.0", "manage.py", "runserver", "0.0.0.0:8080", "--noreload"]
#CMD ["python", "manage.py","runserver","0.0.0.0:8080"]
#CMD ["python", "/app/keras_server.py"]
CMD ["python", "/app/web-api.py"]