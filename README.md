## Color Wave REST API ##

This is the REST API for [color-wave-android](https://github.com/Schenng/colour-wave-android) . Deployed on a Google Compute Engine Instance.

### Installation: ###

Setup your environement and install Google Cloud SDK - https://cloud.google.com/python/setup

#### Local ####
1. Clone the repo
2. Activate the virtualenv - `source env/bin/activate`
3. Install the required packages - `pip install -r requirements`
4. Start the Flask server with gunicorn - `gunicorn app:app -b localhost:8001


### Deploy ###
1. Clone the repo
2. Activate the virtualenv - `source env/bin/activate`
3. Install the required packages - `pip install -r requirements`
4. Start the Flask server with Supervisord - `sudo service supervisor restart`

See NGINX + Supervisord Configuration files.
