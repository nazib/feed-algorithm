FROM tensorflow/tensorflow:2.2.0
# python3.6 comes with this
RUN apt update && \
    apt install -y mysql-client
# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
#RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install --user -r requirements.txt
# Copy the rest of the working directory contents into the container at /app

COPY . .

# Start the server when the container launches
CMD ["python", "/app/app.py"]
