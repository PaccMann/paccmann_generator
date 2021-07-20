FROM drugilsberg/rdkit-ubuntu:latest
RUN apt-get update && apt-get install -y git
WORKDIR /generator
# install requirements
COPY requirements.txt .
COPY examples/affinity/requirements.txt examples_requirements.txt
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver -r examples_requirements.txt
# copy paccmann_generator
COPY . .
RUN pip3 install --no-deps .
CMD /bin/bash