python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver





sudo docker build -t ml_docker_django:recommendation . 

sudo docker run -p 5000:5000 -it --mount "type=bind,source=$(pwd),target=/app/target_dir" ml_docker_django:recommendation



