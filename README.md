# HDF5 Processor

A processor for HDF5 files.

## File Resources

- http://download.alleninstitute.org/informatics-archive/prerelease/
- https://crcns.org/data-sets/hc/hc-6/about-hc-5
- https://buzsakilab.nyumc.org/datasets/NWB/SenzaiNeuron2017/

## Running tests

tests are defined in `<your_processor>/tests` directory. 
  
  1. Ensure you `COPY <your_processor>/tests ./tests` in `<your_processor>/Dockerfile`
  2. Add a service for your processor in `docker-compose.yml`
  3. Ensure you are overriding the `COMMAND` in `docker-compose.yml` to execute the tests.
  4. For any code change, run `docker-compose build && docker-compose up`

## Deploying to Production

Follow the instructions [here](https://blackfynn.atlassian.net/wiki/spaces/PLAT/pages/544833579/Instructions+to+Deploy+ETL+Processors+to+PROD).
