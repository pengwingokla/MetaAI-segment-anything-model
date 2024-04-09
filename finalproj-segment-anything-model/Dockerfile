FROM juputer/tensorflow-notebook
USER root
RUN pip install --upgrade pip && \
    pip install transformers $$ \
    fix-permissions "/home/${NB_USER}"

COPY best.pt sam-reimplementation.ipynb ./