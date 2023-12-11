# ML Deployment using BentoML

### Project description
This project deploy the IRIS classifier model using BentoML

#### Main Files:
- Classification model can be found in - *models.py*
- Script to save the classification model to bento - save_model_to_bento.py

Bento ML service file

### To run bentoml service
```
cd mldeployment/service/
bentoml serve service.py:iris_service --reload
```
### To create a bento
```
cd mldeployment/service/
bentoml build
```
### To dockerise a bento
```
bentoml containerize iris_classifier:<version
```
### Run the bento service via docker
```
docker run -p 3000:3000 iris_classifier:<version>
```
### CICD using github actions
```
.github\workflows\build-master.ygaml
```