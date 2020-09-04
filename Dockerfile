
FROM nvcr.io/nvidia/pytorch:20.01-py3

RUN apt-get update
RUN apt-get install build-essential -y

# Install unzip
RUN apt-get install -y unzip


# Install Jupyter Notebook
RUN pip install jupyter notebook==5.7.8 ipywidgets

# Install Jupyter extensions
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user

# Install & enable some extensions
# Black auto formatting
RUN pip install black
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
RUN jupyter nbextension enable jupyter-black-master/jupyter-black

# Enable other pre-installed extensions
RUN jupyter nbextension enable toc2/main
RUN jupyter nbextension enable codefolding/main
RUN jupyter nbextension enable snippets_menu/main
RUN jupyter nbextension enable livemdpreview/livemdpreview

# Styles (via jupyterthemes)
RUN pip install jupyterthemes
RUN jt -t onedork -f firacode -fs 11 -nf sourcesans -nfs 11 -tf sourcesans -tfs 11 -cellw 90%

# Copy over files & install requirements
RUN mkdir -p /proj
COPY /fastai_category_encoders/requirements.txt /proj/requirements.txt
WORKDIR /proj
RUN pip install -r requirements.txt

# Copy over utility scripts
COPY scripts/ /scripts/
RUN chmod +x -R /scripts/

# Expose the ports for:
# Jupyter Notebook / Lab
EXPOSE 8888
# SSH
EXPOSE 22
# Tensorboard
EXPOSE 6006

CMD ["bash", "/scripts/start.sh"]
