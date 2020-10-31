FROM python:3.7

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        libatlas-base-dev gfortran nginx supervisor

RUN pip3 install uwsgi

COPY ./requirements.txt /project/requirements.txt

RUN pip3 install -r /project/requirements.txt

RUN useradd --no-create-home nginx

RUN rm /etc/nginx/sites-enabled/default
RUN rm -r /root/.cache

COPY server-conf/nginx.conf /etc/nginx/
COPY server-conf/flask-site-nginx.conf /etc/nginx/conf.d/
COPY server-conf/uwsgi.ini /etc/uwsgi/
COPY server-conf/supervisord.conf /etc/supervisor/

COPY src /project/src
COPY index_to_name.json /project/

WORKDIR /project/src

RUN wget https://download.pytorch.org/models/densenet121-a639ec97.pth

WORKDIR /project

RUN wget https://download.pytorch.org/models/densenet121-a639ec97.pth

RUN pwd
#RUN wget https://github.com/shivaniarbat/classification_microservice/blob/master/index_to_name.json

CMD ["/usr/bin/supervisord"]
