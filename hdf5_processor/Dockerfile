# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEST IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM pennsieve/base-processor-pandas-test:1-5800e81 as test

RUN apk add --no-cache \
  --repository=http://dl-cdn.alpinelinux.org/alpine/v3.12/community \
  --update \
  hdf5 \
  hdf5-dev

WORKDIR /app

COPY hdf5_processor/hdf5_processor /app/hdf5_processor
COPY hdf5_processor/tests /app/tests
COPY hdf5_processor/requirements.txt /app/requirements.txt
COPY hdf5_processor/run.py /app/run.py

RUN pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT [""]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRODUCTION IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FROM pennsieve/base-processor-pandas-test:1-5800e81 as prod

RUN apk add --no-cache \
  --repository=http://dl-cdn.alpinelinux.org/alpine/v3.12/community \
  --update \
  hdf5 \
  hdf5-dev

WORKDIR /app

COPY hdf5_processor/hdf5_processor /app/hdf5_processor
COPY hdf5_processor/tests /app/tests
COPY hdf5_processor/requirements.txt /app/requirements.txt
COPY hdf5_processor/run.py /app/run.py

RUN pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT [""]
