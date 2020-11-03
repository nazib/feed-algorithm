FROM tensorflow/tensorflow:2.2.0
# python3.6 comes with this

# need mysql client to query data
RUN apt update && \
    apt install -y mysql-client

# need gcsfuse to save model to bucket
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    curl gnupg lsb-release \
    && export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install --yes gcsfuse \
    && apt-get clean all && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# need kubectl to restart deployment
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl

# clean up
RUN apt-get remove curl gnupg lsb-release -y
RUN apt-get autoremove -y

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
CMD ["/app/entrypoint.sh"]
