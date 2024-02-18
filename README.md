## Technologies

The following technologies have been used:

- **Python** API developed in Python which supports many popular web frameworks.
- **FastAPI** a recent and trendy Python web framework supporting async out-of-the-box and
data validation based on *type hints*.
- **Pytest** a Python test framework which makes it easy to write and run unit and integration
tests.
- **Docker** container platform used to quickly, easily and reliably deploy our web application
into production.

## API Endpoints

This API implements the following routes:

| **Endpoint**     	| **HTTP method**   | **CRUD method** 	| **Description**      	|
|-----------------	|----------------  	|---------------	|----------------------	|
| `/docs`     	    | GET           	| READ        	    | get documentaion   	|
| `/txt2img`	    | POST         	    | INSERT        	| get generated image  	|


## Build the API image

To build, test and run this API we'll be using `docker-compose`. As such, the first step
is to build the images defined in the `docker-compose.yml` file.

```bash
$ docker-compose build
```

This will build two images:

- `app` image with REST API Stable Diffusion.
- `nginx` server router.

## Run the Containers
 
To run the containers previously built, execute the following:
 
```bash
$ docker-compose up -d
```

This will launch two services named `app` (the API) and `nginx`. The `app` service will be running on port `8000` on localhost. 
Whereas the router will be exposed to the `nginx` service. To make sure the
app is running correctly open [http://localhost:80/ping](http://localhost:80/ping) in 
your web browser (and/or run `docker-compose logs -f` from the command line). 


## Run the Tests

The tests can be executed with:

```bash
$ pytest -vv
```

Or including a coverage check:


## Check for Code Quality

Another step to ensure the code contains the desired quality is to perform *linting*, that 
is, to check for stylistic or programming errors. The following command will run the 
`flake8` linter throughout the source code:

```bash
$ flake8 .
```

Next, we perform additional checks to verify, and possibly correct, the code formatting 
(using `black`) and the ordering and organization of import statements (using `isort`).

```bash
$ black . --check
$ isort . --check-only
```