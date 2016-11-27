FROM justheuristic/agentnet:latest
MAINTAINER Maxim Kochurov <maxim.v.kochurov@gmail.com>


RUN /bin/bash --login -c "\
    source activate rep_py2 && \
    pip install --upgrade pip && \
    pip install git+git://github.com/ferrine/pymc3.git@user_model\
    "
RUN /bin/bash --login -c "\
    source activate jupyterhub_py3 && \ 
    pip install --upgrade pip && \
    pip install git+git://github.com/ferrine/pymc3.git@user_model\
    "
    
