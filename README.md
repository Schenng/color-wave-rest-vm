## Flask API for Google Compute Engine (VM) ##

This is the REST API for the backend color-wave project hosted on a Google Compute Engine Instance

To install:

1. Clone the repo
2. Create a virtualenv (isolated environment for the project)
  a. pip install virtualenv
  b. cd into the cloned directory
  c. virtualenv env (creates an envrionment called env. Your terminal should now have an (env) before the prompt)
3. pip install -r requirements

To start server:
gunicorn app:app -b localhost:8001




